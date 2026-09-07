from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import httpx

from decision_engine import EditCandidate

logger = logging.getLogger("ai_suggester.experimental")


D_PROMPT = """Ты — консервативный корректор русского официально-делового текста.
Исправляй только реальные ошибки: орфография, опечатки, пунктуация,
согласование и управление.
Не меняй смысл, правильные слова, термины, аббревиатуры, имена и цифры.
Не улучшай стиль и не переписывай предложения.
Верни только исправленный текст.
"""

F_PROMPT = """Исходный текст:
{TEXT}

Отредактируй исходный текст, исправив ошибки.
Исправляй только орфографию, пунктуацию и регистр.
Не меняй слова и смысл без необходимости.
Верни только исправленный текст без пояснений.
"""

_WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+(?:[-/][А-Яа-яЁёA-Za-z0-9]+)*")


def _words(text: str) -> list[str]:
    return [m.group(0) for m in _WORD_RE.finditer(text)]


def _validate_candidates(candidates: list[EditCandidate]) -> list[EditCandidate]:
    result: list[EditCandidate] = []
    for item in candidates:
        if not item.before or not item.after or item.before == item.after:
            continue
        if "\n" in item.before or "\n" in item.after:
            continue
        if len(item.before) > 80 or len(item.after) > 80:
            continue
        if len(_words(item.before)) > 4 or len(_words(item.after)) > 4:
            continue
        result.append(item)
    return result


def _safe_diff_candidates(source: str, corrected: str, category: str) -> list[EditCandidate]:
    if not source or not corrected or source == corrected:
        return []
    result: list[EditCandidate] = []
    sm = SequenceMatcher(None, source, corrected, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            continue
        before = source[i1:i2]
        after = corrected[j1:j2]
        if not before or not after or "\n" in before or "\n" in after:
            continue
        bw = _words(before)
        aw = _words(after)
        if not bw or not aw or len(bw) != len(aw) or len(bw) > 2:
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
    blocks = [text.strip()]
    blocks.extend(re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE))
    for block in blocks:
        try:
            payload = json.loads(block)
        except Exception:
            continue
        raw = payload.get("edits", []) if isinstance(payload, dict) else []
        if not isinstance(raw, list):
            continue
        candidates: list[EditCandidate] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            before = item.get("before")
            after = item.get("after")
            if not isinstance(before, str) or not isinstance(after, str):
                continue
            try:
                confidence = float(item.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            candidates.append(
                EditCandidate(
                    before=before,
                    after=after,
                    confidence=max(0.0, min(1.0, confidence)),
                    category=str(item.get("category", "unknown")),
                    reason=str(item.get("reason", "")),
                )
            )
        validated = _validate_candidates(candidates)
        if validated:
            return validated
    return []


@dataclass(frozen=True)
class BackendConfig:
    preset: str
    model: str
    base_model: str | None = None
    adapter: str | None = None


class ExperimentalBackend:
    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self._tokenizer = None
        self._model = None
        self._loaded = False

    @property
    def model(self):
        return self._model

    def _device(self):
        assert self._model is not None
        return next(self._model.parameters()).device

    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        import transformers
        from transformers import AutoTokenizer

        logger.info(
            "Experimental[%s]: loading model=%s transformers=%s",
            self.config.preset,
            self.config.model,
            transformers.__version__,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.base_model or self.config.model)
        kwargs: dict[str, Any] = {
            "device_map": os.getenv("EXPERIMENTAL_DEVICE_MAP", "auto"),
            "low_cpu_mem_usage": True,
        }
        dtype = os.getenv("EXPERIMENTAL_DTYPE", "auto")
        if dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, dtype)

        if self.config.preset == "D":
            cls = getattr(transformers, "Qwen3_5ForCausalLM", None)
            if cls is None:
                raise RuntimeError(
                    f"D requires Transformers with Qwen3_5ForCausalLM support; installed={transformers.__version__}"
                )
            self._model = cls.from_pretrained(self.config.base_model or self.config.model, **kwargs)
            self._load_adapter()
        else:
            from transformers import AutoModelForCausalLM
            self._model = AutoModelForCausalLM.from_pretrained(self.config.model, **kwargs)

        self._model.eval()
        self._loaded = True
        logger.info("Experimental[%s]: model ready", self.config.preset)

    def _load_adapter(self) -> None:
        assert self._model is not None
        from peft import PeftModel
        adapter = self.config.adapter
        subfolder = os.getenv("D_ADAPTER_SUBFOLDER", "v4_qwen35_4b_lorugec")
        logger.info("Experimental[D]: loading adapter=%s subfolder=%s", adapter, subfolder)
        self._model = PeftModel.from_pretrained(
            self._model,
            adapter,
            subfolder=subfolder,
            is_trainable=False,
        )
        self._model.eval()
        logger.info("Experimental[D]: adapter loaded")

    def _generate(self, text: str, max_new_tokens: int = 384) -> str:
        self._load()
        import torch
        assert self._tokenizer is not None and self._model is not None

        if self.config.preset == "F":
            prompt = F_PROMPT.format(TEXT=text)
            inputs = self._tokenizer(prompt, return_tensors="pt")
        else:
            messages = [
                {"role": "system", "content": D_PROMPT},
                {"role": "user", "content": f"ИСХОДНЫЙ ТЕКСТ:\n{text}"},
            ]
            try:
                inputs = self._tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    enable_thinking=False,
                )
            except TypeError:
                inputs = self._tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                )

        device = self._device()
        if hasattr(inputs, "items"):
            inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
            input_ids = inputs["input_ids"]
            with torch.inference_mode():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.03,
                )
        else:
            inputs = inputs.to(device)
            input_ids = inputs
            with torch.inference_mode():
                output = self._model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.03,
                )

        generated = output[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def warmup(self) -> None:
        self._load()
        enabled = os.getenv("EXPERIMENTAL_GENERATION_WARMUP", "false").lower() in {"1", "true", "yes", "on"}
        if enabled:
            self._generate("Контрольный текст без ошибок.", 8)
            logger.info("Experimental[%s]: generation warmup OK", self.config.preset)
        else:
            logger.info("Experimental[%s]: model load OK", self.config.preset)

    def candidates(self, raw_text: str) -> list[EditCandidate]:
        generated = self._generate(raw_text)
        parsed = _parse_model_json(generated)
        if parsed:
            return parsed
        return _safe_diff_candidates(raw_text, generated, "specialized-gec")


class LocalEditTagger:
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
        return [
            EditCandidate(
                before=error.before,
                after=error.suggestion,
                confidence=0.88,
                category=error.kind,
                reason=error.explanation,
            )
            for error in errors
            if error.before and error.suggestion and error.before != error.suggestion
        ]


async def verify_with_tlite(raw_text: str, candidates: list[EditCandidate]) -> list[EditCandidate]:
    if not candidates:
        return []
    payload = {
        "model": os.getenv("G_VERIFIER_MODEL", "t-tech/T-lite-it-2.1:q4_K_M"),
        "messages": [
            {
                "role": "system",
                "content": "Ты валидатор предложенных правок русского текста. Не исправляй текст сам. Для каждой правки верни true или false. Принимай только очевидную ошибку.",
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
            "properties": {"accept": {"type": "array", "items": {"type": "boolean"}}},
            "required": ["accept"],
            "additionalProperties": False,
        },
        "think": False,
        "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 128, "repeat_penalty": 1.0},
    }
    try:
        url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        timeout = float(os.getenv("G_VERIFIER_TIMEOUT", "90"))
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{url}/api/chat", json=payload)
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "{}")
        flags = json.loads(content).get("accept", [])
        return [candidate for i, candidate in enumerate(candidates[:12]) if i < len(flags) and bool(flags[i])]
    except Exception as exc:
        logger.warning("G verifier unavailable, rejecting local candidates: %s", exc)
        return []


class ExperimentalRouter:
    def __init__(self, preset: str, morph_detector) -> None:
        self.preset = preset
        self.tagger = LocalEditTagger(morph_detector)
        self.backend: ExperimentalBackend | None = None
        if preset == "D":
            self.backend = ExperimentalBackend(
                BackendConfig(
                    preset="D",
                    model=os.getenv("D_BASE_MODEL", "Qwen/Qwen3.5-4B"),
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
        if self.backend is not None:
            return await asyncio.to_thread(self.backend.candidates, raw_text)
        local = self.tagger.candidates(raw_text)
        if self.preset == "G":
            return await verify_with_tlite(raw_text, local)
        return local

    async def warmup(self) -> None:
        if self.backend is None:
            logger.info("Experimental[%s]: local backend ready", self.preset)
            return
        await asyncio.to_thread(self.backend.warmup)
