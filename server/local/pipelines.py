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

A_SYSTEM = """Ты — строгий корректор русского официально-делового текста.

Ищи только явные ошибки: опечатки, орфографию, пунктуацию, очевидное
согласование и управление. Не улучшай стиль, не перефразируй, не меняй
термины, аббревиатуры, имена, названия, номера, даты и допустимые словоформы.
Не нормализуй е/ё.

Возвращай только локальные edits. BEFORE должен быть точной непрерывной
подстрокой исходного текста. AFTER — минимальная замена только этого фрагмента.
Не возвращай исправленный абзац целиком. При сомнении edits=[]. Максимум 12.

Важно: «изучена» → «изучено», «должностного» → «должностных» и
«деятельностей» → «деятельности» без явного грамматического основания
не являются допустимыми правками.
"""

G_SYSTEM = """Ты — консервативный валидатор предложенных правок русского
официально-делового текста. Не исправляй текст сам и не предлагай новые правки.
Для каждого кандидата ответь true только если ошибка очевидна по предложению
и вариант AFTER однозначен. Валидные, но альтернативные формы отклоняй.
При сомнении — false.
"""

F_PROMPT = """Исходный текст:
{TEXT}

Отредактируй исходный текст, исправив только явные ошибки.
Не перестраивай предложения, не меняй порядок слов и не изменяй грамматические
формы слов без однозначного основания. Сохрани абзацы и переносы строк.
"""


@dataclass(frozen=True)
class StackInfo:
    name: str
    description: str
    model: str
    experimental: bool


STACKS: dict[str, StackInfo] = {
    "A": StackInfo("A", "production: T-lite + structured edits + deterministic rescue", A_MODEL, False),
    "F": StackInfo("F", "experimental: Spell-Corrector-RU-4B + surface-only gate", F_MODEL, True),
    "G": StackInfo("G", "experimental: morphology rescue + T-lite verifier", G_MODEL, True),
}


class SurfaceGate:
    """Reject surface-model edits that alter grammatical word features."""

    def __init__(self) -> None:
        try:
            import pymorphy3

            self.morph = pymorphy3.MorphAnalyzer()
            self.available = True
        except Exception as exc:  # pragma: no cover
            self.morph = None
            self.available = False
            logger.warning("SurfaceGate: pymorphy3 unavailable: %s", exc)

    def _signature(self, word: str) -> tuple[str, ...] | None:
        if self.morph is None:
            return None
        parses = self.morph.parse(word)
        if not parses:
            return None
        tag = parses[0].tag
        attrs = ("POS", "case", "number", "gender", "tense", "person", "voice", "aspect", "mood", "animacy")
        return tuple(f"{name}={getattr(tag, name, None)}" for name in attrs)

    def accept(self, before: str, after: str) -> bool:
        if not before or not after or before == after or "\n" in before or "\n" in after:
            return False
        bw = WORD_RE.findall(before)
        aw = WORD_RE.findall(after)
        if len(bw) != len(aw):
            return False
        if not bw:
            return True
        if not self.available:
            return all(x.casefold() == y.casefold() for x, y in zip(bw, aw))
        return all(self._signature(x) == self._signature(y) for x, y in zip(bw, aw))


def surface_candidates(source: str, corrected: str) -> list[EditCandidate]:
    if source == corrected or not source or not corrected:
        return []
    if source.count("\n") != corrected.count("\n"):
        return []
    if len(source.split("\n\n")) != len(corrected.split("\n\n")):
        return []

    out: list[EditCandidate] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, source, corrected, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            before, after = source[i1:i2], corrected[j1:j2]
            if 0 < len(before) <= 80 and 0 < len(after) <= 80 and "\n" not in before and "\n" not in after:
                out.append(EditCandidate(before, after, 0.80, "surface-model", "bounded surface diff"))
            continue
        left = max(0, i1 - 12)
        right = min(len(source), i2 + 12)
        left_b = max(0, j1 - 12)
        right_b = min(len(corrected), j2 + 12)
        before, after = source[left:right], corrected[left_b:right_b]
        if len(before) <= 80 and len(after) <= 80 and "\n" not in before and "\n" not in after:
            out.append(EditCandidate(before, after, 0.78, "surface-model", "punctuation context diff"))
    return out


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
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": schema,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx, "num_predict": self.num_predict, "num_thread": self.num_threads, "repeat_penalty": 1.03},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "{}")
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    async def candidates(self, text: str, context: str = "") -> list[EditCandidate]:
        user = f"Контекст документа:\n{context}\n\n" if context else ""
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
        return DecisionEngine.parse(await self.chat_json(A_SYSTEM, user, schema))

    async def verify(self, text: str, candidates: list[EditCandidate]) -> list[EditCandidate]:
        if not candidates:
            return []
        payload = [{"id": i, "before": c.before, "after": c.after, "category": c.category} for i, c in enumerate(candidates[:12])]
        schema = {
            "type": "object",
            "properties": {"accept": {"type": "array", "maxItems": 12, "items": {"type": "boolean"}}},
            "required": ["accept"],
            "additionalProperties": False,
        }
        data = await self.chat_json(G_SYSTEM, json.dumps({"text": text, "candidates": payload}, ensure_ascii=False), schema)
        flags = data.get("accept") if isinstance(data, dict) else None
        if not isinstance(flags, list):
            return []
        return [c for i, c in enumerate(candidates[:12]) if i < len(flags) and bool(flags[i])]


class MorphologyRescue:
    """Small deterministic rescue for adjacent adjective/participle + noun."""

    def __init__(self) -> None:
        try:
            import pymorphy3

            self.morph = pymorphy3.MorphAnalyzer()
            self.available = True
        except Exception as exc:  # pragma: no cover
            self.morph = None
            self.available = False
            logger.warning("MorphologyRescue: pymorphy3 unavailable: %s", exc)

    @staticmethod
    def _is_adj(parse: Any) -> bool:
        tag = str(parse.tag)
        return any(x in tag for x in ("ADJF", "ADJS", "PRTF", "PRTS"))

    @staticmethod
    def _is_noun(parse: Any) -> bool:
        return "NOUN" in str(parse.tag)

    def candidates(self, text: str) -> list[EditCandidate]:
        if not self.available or self.morph is None:
            return []
        tokens = list(WORD_RE.finditer(text))
        result: list[EditCandidate] = []
        for left, right in zip(tokens, tokens[1:]):
            if left.end() == right.start():
                continue
            adjective = left.group(0)
            noun = right.group(0)
            apos = [p for p in self.morph.parse(adjective) if self._is_adj(p) and p.tag.number and p.tag.case]
            npos = [p for p in self.morph.parse(noun) if self._is_noun(p)]
            if not apos or not npos:
                continue
            compatible = any(
                a.tag.number == n.tag.number and a.tag.case == n.tag.case and (
                    not a.tag.gender or not n.tag.gender or a.tag.gender == n.tag.gender
                )
                for a in apos for n in npos
            )
            if compatible:
                continue
            best = apos[0]
            noun_parse = npos[0]
            if any(p.tag.case == "ablt" for p in npos) and any("PRTF" in str(p.tag) or "PRTS" in str(p.tag) for p in apos):
                continue
            inflected = noun_parse.inflect({g for g in (best.tag.number, best.tag.case) if g})
            if not inflected or inflected.word == noun:
                continue
            after = inflected.word
            if noun[:1].isupper():
                after = after[:1].upper() + after[1:]
            result.append(EditCandidate(
                before=noun,
                after=after,
                confidence=0.90,
                category="agreement",
                reason="явное рассогласование прилагательного и существительного",
            ))
        return result


class FBackend:
    def __init__(self) -> None:
        self.model_id = F_MODEL
        self._tokenizer = None
        self._model = None
        self.surface_gate = SurfaceGate()

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Experimental[F]: loading model=%s", self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=os.getenv("EXPERIMENTAL_DEVICE_MAP", "auto"),
            low_cpu_mem_usage=True,
        )
        self._model.eval()
        logger.info("Experimental[F]: model ready")

    def _device(self):
        assert self._model is not None
        return next(self._model.parameters()).device

    def correct(self, text: str) -> list[EditCandidate]:
        self._load()
        import torch

        assert self._tokenizer is not None and self._model is not None
        prompt = F_PROMPT.format(TEXT=text)
        encoded = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = self._device()
        inputs = {k: v.to(device) for k, v in encoded.items() if torch.is_tensor(v)}
        input_ids = inputs["input_ids"]
        max_new = min(192, max(64, len(text) // 2 + 32))
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=0.1,
                top_p=0.7,
                repetition_penalty=1.02,
            )
        generated = self._tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
        raw = surface_candidates(text, generated)
        safe = [c for c in raw if self.surface_gate.accept(c.before, c.after)]
        logger.info("Experimental[F]: generated_chars=%d raw_candidates=%d safe_candidates=%d", len(generated), len(raw), len(safe))
        return safe


class LocalEditTagger:
    def __init__(self, detector: Any) -> None:
        self.detector = detector
        self.rescue = MorphologyRescue()

    def candidates(self, text: str) -> list[EditCandidate]:
        out: list[EditCandidate] = []
        if self.detector is not None and getattr(self.detector, "available", False):
            try:
                errors = self.detector.detect_errors(text)
                out.extend(
                    EditCandidate(e.before, e.suggestion, 0.90, e.kind, e.explanation)
                    for e in errors
                    if e.before and e.suggestion and e.before != e.suggestion
                )
            except Exception as exc:
                logger.warning("LocalEditTagger failed: %s", exc)
        out.extend(self.rescue.candidates(text))
        dedup: dict[tuple[str, str], EditCandidate] = {}
        for candidate in out:
            dedup.setdefault((candidate.before, candidate.after), candidate)
        return list(dedup.values())[:12]


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
                {"type": "object", "properties": {"edits": {"type": "array"}}, "required": ["edits"], "additionalProperties": False},
            )

    @staticmethod
    def _merge_candidates(*groups: list[EditCandidate]) -> list[EditCandidate]:
        result: dict[tuple[str, str], EditCandidate] = {}
        for group in groups:
            for c in group:
                result.setdefault((c.before, c.after), c)
        return list(result.values())[:12]

    async def candidates(self, text: str, context: str = "") -> list[EditCandidate]:
        if self.preset == "A":
            llm = await self.tlite.candidates(text, context)
            rescue = self.tagger.candidates(text)
            merged = self._merge_candidates(llm, rescue)
            logger.info("Stack A candidates: llm=%d rescue=%d merged=%d", len(llm), len(rescue), len(merged))
            return merged
        if self.preset == "G":
            local = self.tagger.candidates(text)
            verified = await self.tlite.verify(text, local)
            logger.info("Stack G candidates: local=%d verified=%d", len(local), len(verified))
            return verified
        return await asyncio.to_thread(self.f_backend.correct, text)  # type: ignore[union-attr]
