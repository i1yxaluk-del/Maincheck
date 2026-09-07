from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import warnings
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import httpx

from decision_engine import EditCandidate

logger = logging.getLogger("ai_suggester.experimental")


GEC_PROMPT = """Ты — консервативный корректор русского официально-делового текста.
Найди только реальные ошибки в исходном тексте: орфография, опечатки,
пунктуация, согласование, управление.

КРИТИЧЕСКИЕ ПРАВИЛА:
- не переписывай предложения целиком;
- не меняй правильные слова, имена, цифры, термины, названия и аббревиатуры;
- не улучшай стиль;
- не заменяй допустимую словоформу на другую допустимую словоформу;
- каждое BEFORE должно быть точным фрагментом исходного текста;
- предлагай минимальную локальную замену;
- если не уверен — не предлагай правку.

Верни только JSON:
{"edits":[{"before":"...","after":"...","confidence":0.0,"category":"...","reason":"..."}]}
"""


@dataclass(frozen=True)
class BackendConfig:
    preset: str
    model: str
    base_model: str | None = None
    adapter: str | None = None


_WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+(?:[-/][А-Яа-яЁёA-Za-z0-9]+)*")


def _words(text: str) -> list[str]:
    return [m.group(0) for m in _WORD_RE.finditer(text)]


def _is_wordless(text: str) -> bool:
    return bool(text) and not _WORD_RE.search(text)


def _safe_diff_candidates(source: str, corrected: str, category: str) -> list[EditCandidate]:
    if not source or not corrected or source == corrected:
        return []
    sm = SequenceMatcher(None, source, corrected, autojunk=False)
    result: list[EditCandidate] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        before = source[i1:i2]
        after = corrected[j1:j2]
        if not before or not after or before == after:
            continue
        bw = _words(before)
        aw = _words(after)

        # Surface models are never allowed to turn paragraph reformatting into
        # lexical corrections. Ignore whitespace/newline-only changes completely.
        if not bw or not aw:
            continue
        # Do not accept insertions/deletions or large rewrites from a generated
        # fallback; only a local 1-2 word substitution is eligible.
        if len(bw) != len(aw) or len(bw) > 2:
            continue
        if "\n" in before or "\n" in after:
            continue
        result.append(
            EditCandidate(
                before=before,
                after=after,
                confidence=0.80,
                category=category,
                reason="safe local diff from specialized model",
            )
        )
    return result


def _parse_model_json(text: str) -> list[EditCandidate]:
    if not text:
        return []
    candidates: list[EditCandidate] = []
    blocks = [text.strip()]
    blocks.extend(
        re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    )
    for block in blocks:
        try:
            payload = json.loads(block)
        except Exception:
            continue
        raw = payload.get("edits", []) if isinstance(payload, dict) else []
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            before = item.get("before")
            after = item.get("after")
            if not isinstance(before, str) or not isinstance(after, str):
                continue
            try:
                conf = float(item.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            candidates.append(
                EditCandidate(
                    before=before,
                    after=after,
                    confidence=max(0.0, min(1.0, conf)),
                    category=str(item.get("category", "unknown")),
                    reason=str(item.get("reason", "")),
                )
        if candidates:
            return _validate_candidates(candidates)
    return []


def _validate_candidates(candidates: list[EditCandidate]) -> list[EditCandidate]:
    out: list[EditCandidate] = []
    for candidate in candidates:
        if not candidate.before or not candidate.after or candidate.before == candidate.after:
            continue
        # Candidate source fragments must stay local even if the model returns a
        # syntactically valid JSON object describing a rewrite of a whole phrase.
        if len(candidate.before) > 80 or len(candidate.before.split()) > 4:
            continue
        if len(candidate.after.split()) > 4:
            continue
        if "\n" in candidate.before or "\n" in candidate.after:
            continue
        out.append(candidate)
    return out


class ExperimentalBackend:
    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self._tokenizer = None
        self._model = None
        self._loaded = False

    def _load_transformers(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "D/F require the common requirements.txt stack: transformers, torch, accelerate and peft"
            ) from exc

        base_model = self.config.base_model or self.config.model
        device_map = os.getenv("EXPERIMENTAL_DEVICE_MAP", "auto")
        dtype = os.getenv("EXPERIMENTAL_DTYPE", "auto")
        logger.info(
            "Experimental[%s]: loading base=%s device_map=%s dtype=%s transformers=%s",
            self.config.preset,
            base_model,
            device_map,
            dtype,
            transformers.__version__,
        )

        if self.config.preset == "D" and not hasattr(transformers, "Qwen3_5ForCausalLM"):
            raise RuntimeError(
                "D requires Transformers with Qwen3_5ForCausalLM support. "
                f"Installed transformers={transformers.__version__}; run the one-time installer to install 5.16.1."
            )

        self._tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        kwargs: dict[str, Any] = {
            "device_map": device_map,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, dtype)

        if self.config.preset == "D":
            # Qwen documents Qwen3_5ForCausalLM as the text-only architecture.
            # Using the explicit class avoids AutoModel dispatching through the
            # multimodal top-level config on older/newer Transformers revisions.
            model_cls = transformers.Qwen3_5ForCausalLM
            self._model = model_cls.from_pretrained(base_model, **kwargs)
        else:
            self._model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)

        if self.config.adapter:
            self._load_adapter_cleanly()

        self._model.eval()
        self._loaded = True
        logger.info("Experimental[%s]: model ready", self.config.preset)

    def _load_adapter_cleanly(self) -> None:
        assert self._model is not None
        adapter_repo = self.config.adapter
        subfolder = os.getenv("D_ADAPTER_SUBFOLDER", "v4_qwen35_4b_lorugec")
        adapter_name = os.getenv("D_ADAPTER_NAME", "gec")
        logger.info(
            "Experimental[D]: loading adapter=%s subfolder=%s",
            adapter_repo,
            subfolder,
        )

        # Prefer Transformers' native PEFT adapter integration. It preserves the
        # Qwen3.5 module naming used by the released adapter and avoids wrapping
        # the model in a second model class unnecessarily.
        if hasattr(self._model, "load_adapter"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    self._model.load_adapter(
                        adapter_repo,
                        adapter_name=adapter_name,
                        subfolder=subfolder,
                        is_trainable=False,
                    )
                except TypeError:
                    self._load_adapter_with_peft(adapter_repo, subfolder)
                    caught = []
            self._raise_on_adapter_warnings(caught)
            if hasattr(self._model, "set_adapter"):
                self._model.set_adapter(adapter_name)
            logger.info("Experimental[D]: adapter loaded via Transformers PEFT integration")
            return

        self._load_adapter_with_peft(adapter_repo, subfolder)
        logger.info("Experimental[D]: adapter loaded via PEFT")

    def _load_adapter_with_peft(self, adapter_repo: str, subfolder: str) -> None:
        assert self._model is not None
        try:
            from peft import PeftModel
        except Exception as exc:
            raise RuntimeError("D requires peft; install the common requirements.txt stack") from exc
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._model = PeftModel.from_pretrained(
                self._model,
                adapter_repo,
                subfolder=subfolder,
                is_trainable=False,
            )
        self._raise_on_adapter_warnings(caught)

    @staticmethod
    def _raise_on_adapter_warnings(caught) -> None:
        adapter_warnings = [
            str(item.message)
            for item in caught
            if "missing adapter keys" in str(item.message).lower()
        ]
        if adapter_warnings:
            raise RuntimeError(
                "D adapter did not load cleanly: PEFT reported missing adapter keys. "
                "The base model, adapter and Transformers/PEFT versions do not match. "
                "Use the versions from requirements-experimental.txt and re-run the one-time model installer."
            )

    def _prepare_inputs(self, text: str):
        """Return a tensor mapping suitable for model.generate(**inputs)."""
        assert self._tokenizer is not None
        messages = [
            {"role": "system", "content": GEC_PROMPT},
            {"role": "user", "content": f"ИСХОДНЫЙ ТЕКСТ:\n{text}"},
        ]
        template_kwargs = {
            "messages": messages,
            "add_generation_prompt": True,
            "tokenize": True,
            "return_tensors": "pt",
            "return_dict": True,
        }
        try:
            inputs = self._tokenizer.apply_chat_template(
                enable_thinking=False,
                **template_kwargs,
            )
        except TypeError:
            inputs = self._tokenizer.apply_chat_template(**template_kwargs)
        if hasattr(inputs, "items"):
            target_device = self._model.device
            return {k: v.to(target_device) for k, v in inputs.items() if hasattr(v, "to")}
        return inputs.to(self._model.device)

    def _generate(self, text: str, max_new_tokens: int = 256) -> str:
        self._load_transformers()
        import torch

        assert self._tokenizer is not None and self._model is not None
        inputs = self._prepare_inputs(text)
        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs
        with torch.inference_mode():
            if isinstance(inputs, dict):
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    top_p=1.0,
                    repetition_penalty=1.03,
                )
            else:
                output = self._model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    top_p=1.0,
                    repetition_penalty=1.03,
                )
        generated = output[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def warmup(self) -> None:
        """Load model and optionally run a tiny deterministic generation before serving."""
        if not self._loaded:
            self._load_transformers()
        generation_warmup = os.getenv(
            "EXPERIMENTAL_GENERATION_WARMUP",
            "false" if self.config.preset == "D" else "true",
        ).lower() in {"1", "true", "yes", "on"}
        if generation_warmup:
            _ = self._generate("Контрольный текст без ошибок.", max_new_tokens=8)
            logger.info("Experimental[%s]: warmup OK", self.config.preset)
        else:
            logger.info("Experimental[%s]: model load OK (generation warmup disabled)", self.config.preset)

    def candidates(self, raw_text: str) -> list[EditCandidate]:
        output = self._generate(raw_text)
        parsed = _parse_model_json(output)
        if parsed:
            return parsed
        return _safe_diff_candidates(raw_text, output, "specialized-gec")


class LocalEditTagger:
    """Local token-level candidate generator used by E/G."""

    def __init__(self, morph_detector) -> None:
        self.detector = morph_detector

    def candidates(self, raw_text: str) -> list[EditCandidate]:
        if self.detector is None or not getattr(self.detector, "available", False):
            return []
        try:
            errors = self.detector.detect_errors(raw_text)
        except Exception as exc:
            logger.warning("LocalEditTagger failed: %s", exc)
            return []
        out: list[EditCandidate] = []
        for err in errors:
            if not err.before or not err.suggestion or err.before == err.suggestion:
                continue
            out.append(
                EditCandidate(
                    before=err.before,
                    after=err.suggestion,
                    confidence=0.88,
                    category=err.kind,
                    reason=err.explanation,
                )
            )
        return out


async def verify_with_tlite(raw_text: str, candidates: list[EditCandidate]) -> list[EditCandidate]:
    if not candidates:
        return []
    model = os.getenv("G_VERIFIER_MODEL", "t-tech/T-lite-it-2.1:q4_K_M")
    url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты валидатор предложенных правок русского текста. "
                    "Не исправляй текст сам. Для каждой правки верни true/false. "
                    "Принимай только очевидную ошибку, относящуюся к исходному тексту."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "text": raw_text,
                        "candidates": [
                            {"before": c.before, "after": c.after, "category": c.category}
                            for c in candidates[:12]
                        ],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "stream": False,
        "format": {
            "type": "object",
            "properties": {
                "accept": {"type": "array", "items": {"type": "boolean"}}
            },
            "required": ["accept"],
            "additionalProperties": False,
        },
        "think": False,
        "options": {
            "temperature": 0,
            "num_ctx": 2048,
            "num_predict": 128,
            "repeat_penalty": 1.0,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=float(os.getenv("G_VERIFIER_TIMEOUT", "90"))) as client:
            response = await client.post(f"{url}/api/chat", json=payload)
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "{}")
        data = json.loads(content)
        flags = data.get("accept", [])
        return [c for i, c in enumerate(candidates[:12]) if i < len(flags) and bool(flags[i])]
    except Exception as exc:
        logger.warning("G verifier unavailable, rejecting local candidates: %s", exc)
        return []


class ExperimentalRouter:
    def __init__(self, preset: str, morph_detector) -> None:
        self.preset = preset
        self.morph_tagger = LocalEditTagger(morph_detector)
        self.backend: ExperimentalBackend | None = None
        if preset == "D":
            self.backend = ExperimentalBackend(
                BackendConfig(
                    preset="D",
                    model=os.getenv("D_MODEL", "Qwen/Qwen3.5-4B"),
                    base_model=os.getenv("D_BASE_MODEL", "Qwen/Qwen3.5-4B"),
                    adapter=os.getenv("D_ADAPTER", "synterr-nlp/bea2026-gec-adapters"),
                )
            )
        elif preset == "F":
            self.backend = ExperimentalBackend(
                BackendConfig(
                    preset="F",
                    model=os.getenv("F_MODEL", "melsmm/Spell-Corrector-RU-4B"),
                )
            )

    async def candidates(self, raw_text: str) -> list[EditCandidate]:
        if self.preset in {"D", "F"} and self.backend is not None:
            return await asyncio.to_thread(self.backend.candidates, raw_text)
        local = self.morph_tagger.candidates(raw_text)
        if self.preset == "G":
            return await verify_with_tlite(raw_text, local)
        return local

    async def warmup(self) -> None:
        if self.preset not in {"D", "F"} or self.backend is None:
            logger.info("Experimental[%s]: warmup uses local edit backend (no model load)", self.preset)
            return
        if os.getenv("EXPERIMENTAL_WARMUP", "true").lower() not in {"1", "true", "yes", "on"}:
            logger.info("Experimental[%s]: warmup disabled by EXPERIMENTAL_WARMUP=false", self.preset)
            return
        try:
            await asyncio.to_thread(self.backend.warmup)
        except Exception as exc:
            logger.exception("Experimental[%s]: warmup failed: %s", self.preset, exc)
            raise
