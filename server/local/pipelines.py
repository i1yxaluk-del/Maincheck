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

from decision_engine import DecisionEngine, EditCandidate

logger = logging.getLogger("ai_suggester.pipelines")

A_MODEL = "t-tech/T-lite-it-2.1:q4_K_M"
F_MODEL = os.getenv("F_MODEL", "melsmm/Spell-Corrector-RU-4B")
G_MODEL = os.getenv("G_VERIFIER_MODEL", A_MODEL)

WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z]+(?:[-/][А-Яа-яЁёA-Za-z]+)*")

A_SYSTEM = """Ты — консервативный корректор русского официально-делового текста.

Исправляй только явные ошибки: орфографию, опечатки, пунктуацию,
согласование и управление. Не переписывай стиль и не перефразируй текст.
Не меняй корректные падежи, формы слов, термины, аббревиатуры, имена,
названия организаций, номера, даты и значения.
Не исправляй е/ё.

Работай только как редактор локальных правок. Не возвращай целиком текст.
Для каждой правки указывай точный фрагмент BEFORE из исходного текста.
Если сомневаешься хотя бы немного — правку не предлагай.
Максимум 12 правок.
"""

G_SYSTEM = """Ты — валидатор предложенных правок русского официально-делового текста.
Не исправляй текст сам и не предлагай новых правок.
Для каждой предложенной правки реши только: очевидно ли, что это ошибка.
Принимай только правки с практически нулевым риском ложного срабатывания.
"""

F_PROMPT = """Исходный текст:
{TEXT}

Отредактируй исходный текст, исправив ошибки."""


@dataclass(frozen=True)
class StackInfo:
    name: str
    description: str
    model: str
    experimental: bool


STACKS: dict[str, StackInfo] = {
    "A": StackInfo("A", "production: T-lite + строгий JSON + локальный safety gate", A_MODEL, False),
    "F": StackInfo("F", "experimental: Spell-Corrector-RU-4B + surface-only gate", F_MODEL, True),
    "G": StackInfo("G", "experimental: local edit detector + T-lite verifier", G_MODEL, True),
}


class SurfaceGate:
    """Reject full-text model edits that alter grammatical word forms."""

    def __init__(self) -> None:
        try:
            import pymorphy3

            self.morph = pymorphy3.MorphAnalyzer()
            self.available = True
        except Exception as exc:  # pragma: no cover - environment specific
            self.morph = None
            self.available = False
            logger.warning("SurfaceGate: pymorphy3 unavailable: %s", exc)

    def _signature(self, word: str) -> tuple[str, ...] | None:
        if self.morph is None:
            return None
        parses = self.morph.parse(word)
        if not parses:
            return None
        p = parses[0]
        tag = p.tag
        attrs = (
            "POS", "case", "number", "gender", "tense", "person",
            "voice", "aspect", "mood", "animacy",
        )
        return tuple(f"{name}={getattr(tag, name, None)}" for name in attrs)

    def _tokens_compatible(self, before: str, after: str) -> bool:
        bw = WORD_RE.findall(before)
        aw = WORD_RE.findall(after)
        if len(bw) != len(aw):
            return False
        if not bw:
            return True
        if not self.available:
            # Fail closed: a surface model may not alter alphabetic tokens
            # when we cannot validate their morphology.
            return all(x.casefold() == y.casefold() for x, y in zip(bw, aw))
        return all(self._signature(x) == self._signature(y) for x, y in zip(bw, aw))

    def accept(self, before: str, after: str) -> bool:
        if "\n" in before or "\n" in after:
            return False
        if before == after:
            return False
        return self._tokens_compatible(before, after)


def _char_diff_candidates(source: str, corrected: str, category: str) -> list[EditCandidate]:
    if source == corrected or not source or not corrected:
        return []
    if source.count("\n") != corrected.count("\n"):
        return []
    sm = SequenceMatcher(None, source, corrected, autojunk=False)
    candidates: list[EditCandidate] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            # insert/delete may be punctuation fixes only; keep them in a
            # later bounded-window pass instead of allowing paragraph reflow.
            continue
        before = source[i1:i2]
        after = corrected[j1:j2]
        if not before.strip() or not after.strip():
            continue
        if len(before) > 80 or len(after) > 80:
            continue
        candidates.append(EditCandidate(
            before=before,
            after=after,
            confidence=0.80,
            category=category,
            reason="bounded diff from specialized surface model",
        ))
    return candidates


class TliteClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.model = A_MODEL
        self.timeout = float(os.getenv("OLLAMA_TIMEOUT", "120"))
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
        self.num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "384"))
        self.num_threads = int(os.getenv("NUM_THREADS", "28"))
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "24h")
        self.temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0"))

    async def chat_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": schema,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "num_thread": self.num_threads,
                "repeat_penalty": 1.03,
            },
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "{}")
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    async def candidates(self, text: str, context: str = "") -> list[EditCandidate]:
        user = ""
        if context:
            user += f"Контекст документа:\n{context}\n\n"
        user += f"ТЕКСТ ДЛЯ ПРОВЕРКИ:\n{text}"
        schema = {
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "maxItems": 12,
                    "items": {
                        "type": "object",
                        "properties": {
                            "before": {"type": "string"},
                            "after": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "category": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["before", "after", "confidence", "category", "reason"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["edits"],
            "additionalProperties": False,
        }
        data = await self.chat_json(A_SYSTEM, user, schema)
        return DecisionEngine.parse(data)

    async def verify(self, text: str, candidates: list[EditCandidate]) -> list[EditCandidate]:
        if not candidates:
            return []
        payload_candidates = [
            {"id": idx, "before": c.before, "after": c.after, "category": c.category}
            for idx, c in enumerate(candidates[:12])
        ]
        schema = {
            "type": "object",
            "properties": {
                "accept": {"type": "array", "items": {"type": "boolean", "maxItems": 12}}
            },
            "required": ["accept"],
            "additionalProperties": False,
        }
        user = json.dumps({"text": text, "candidates": payload_candidates}, ensure_ascii=False)
        data = await self.chat_json(G_SYSTEM, user, schema)
        flags = data.get("accept") if isinstance(data, dict) else None
        if not isinstance(flags, list):
            return []
        return [c for idx, c in enumerate(candidates[:12]) if idx < len(flags) and bool(flags[idx])]


class FBackend:
    def __init__(self) -> None:
        self.model_id = F_MODEL
        self._tokenizer = None
        self._model = None
        self.surface_gate = SurfaceGate()

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Experimental[F]: loading model=%s", self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        device_map = os.getenv("EXPERIMENTAL_DEVICE_MAP", "auto")
        kwargs: dict[str, Any] = {"device_map": device_map, "low_cpu_mem_usage": True}
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self._model.eval()
        logger.info("Experimental[F]: model ready")

    def correct(self, text: str) -> list[EditCandidate]:
        self._load()
        import torch

        assert self._tokenizer is not None and self._model is not None
        prompt = F_PROMPT.format(TEXT=text)
        messages = [{"role": "user", "content": prompt}]
        inputs = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self._model.device)
        max_new = min(256, max(64, len(text) // 2 + 32))
        with torch.inference_mode():
            output = self._model.generate(
                inputs,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=0.1,
                top_p=0.7,
                repetition_penalty=1.02,
            )
        generated = self._tokenizer.decode(
            output[0][inputs.shape[1]:], skip_special_tokens=True
        ).strip()
        raw = _char_diff_candidates(text, generated, "surface-model")
        safe = [c for c in raw if self.surface_gate.accept(c.before, c.after)]
        return safe


class LocalEditTagger:
    def __init__(self, detector: Any) -> None:
        self.detector = detector

    def candidates(self, text: str) -> list[EditCandidate]:
        if self.detector is None or not getattr(self.detector, "available", False):
            return []
        try:
            errors = self.detector.detect_errors(text)
        except Exception as exc:
            logger.warning("LocalEditTagger failed: %s", exc)
            return []
        return [
            EditCandidate(
                before=e.before,
                after=e.suggestion,
                confidence=0.90,
                category=e.kind,
                reason=e.explanation,
            )
            for e in errors
            if e.before and e.suggestion and e.before != e.suggestion
        ]


class StackRouter:
    def __init__(self, preset: str, morph_detector: Any) -> None:
        if preset not in STACKS:
            raise RuntimeError(f"Unsupported LLM_PRESET={preset!r}; expected A, F or G")
        self.preset = preset
        self.info = STACKS[preset]
        self.tlite = TliteClient()
        self.tagger = LocalEditTagger(morph_detector)
        self.f_backend = FBackend() if preset == "F" else None

    async def warmup(self) -> None:
        if self.preset == "F":
            await asyncio.to_thread(self.f_backend._load)  # type: ignore[union-attr]
        else:
            await self.tlite.chat_json(
                A_SYSTEM,
                "Контрольный текст без ошибок.",
                {"type": "object", "properties": {"edits": {"type": "array"}}, "required": ["edits"]},
            )

    async def candidates(self, text: str, context: str = "") -> list[EditCandidate]:
        if self.preset == "A":
            return await self.tlite.candidates(text, context)
        if self.preset == "G":
            local = self.tagger.candidates(text)
            return await self.tlite.verify(text, local)
        return await asyncio.to_thread(self.f_backend.correct, text)  # type: ignore[union-attr]
