from __future__ import annotations

import difflib
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx

from decision_engine import DecisionEngine, EditCandidate

logger = logging.getLogger("ai_suggester.hybrid")
WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z]+(?:[-/][А-Яа-яЁёA-Za-z]+)*")
TLITE = "t-tech/T-lite-it-2.1:q4_K_M"
GIGACHAT = "hf.co/ai-sage/GigaChat3.1-10B-A1.8B-GGUF:latest"
QWEN35 = "Qwen/Qwen3.5-4B"

SYSTEM = """Ты — профессиональный редактор русского официально-делового текста.

Исправь только объективные ошибки языка. Анализируй всё предложение и связанные
слова, а не отдельное слово. Проверь управление, согласование, орфографию,
пунктуацию по синтаксическому контексту, числительные, однородные члены,
формы слов после предлогов и глаголов, явные лишние/пропущенные знаки.

Не исправляй стиль, терминологию, названия, аббревиатуры, даты и номера.
Не меняй допустимую словоформу только потому, что другая форма звучит лучше.
Не перестраивай предложения, не объединяй и не разделяй абзацы.
Сохрани все переносы строк и порядок слов, кроме минимального исправления.

Верни только JSON: {\"corrected\":\"...\"}
"""

JUDGE_SYSTEM = """Ты — независимый редактор-контролёр русского официально-делового текста.
Тебе дан исходный текст и локальные кандидаты исправления. Для каждого кандидата
проверь полное предложение, синтаксическую связь и смысл. Одобряй только объективную
ошибку. Не одобряй стилистические замены, допустимые варианты, изменение терминов,
дат и чисел. При сомнении — false.
Верни только JSON: {\"accept\":[true,false,...]}
"""


@dataclass(frozen=True)
class StackInfo:
    name: str
    description: str
    model: str
    experimental: bool


STACKS = {
    "A": StackInfo("A", "production: T-lite draft + GigaChat judge + local/syntax guards", TLITE, False),
    "B": StackInfo("B", "production-candidate: GigaChat draft + T-lite judge + local/syntax guards", GIGACHAT, False),
    "X": StackInfo("X", "experimental: Qwen3.5-4B reasoning draft + cross-model judge", QWEN35, True),
    "Y": StackInfo("Y", "experimental: Qwen3.5 self-consistency + LanguageTool/syntax adjudication", QWEN35, True),
}


class OllamaJSON:
    def __init__(self, model: str) -> None:
        self.url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.model = model
        self.timeout = float(os.getenv("HYBRID_LLM_TIMEOUT", os.getenv("OLLAMA_TIMEOUT", "300")))
        self.ctx = int(os.getenv("HYBRID_NUM_CTX", "4096"))
        self.predict = int(os.getenv("HYBRID_NUM_PREDICT", "512"))
        self.threads = int(os.getenv("NUM_THREADS", "16"))
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

    async def call(self, messages: list[dict[str, str]], schema: dict[str, Any], *, temperature: float = 0.0, think: bool = False) -> dict[str, Any]:
        payload = {
            "model": self.model, "messages": messages, "stream": False, "format": schema,
            "think": think, "keep_alive": self.keep_alive,
            "options": {"temperature": temperature, "num_ctx": self.ctx, "num_predict": self.predict,
                         "num_thread": self.threads, "repeat_penalty": 1.02},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.url}/api/chat", json=payload)
            r.raise_for_status()
            content = r.json().get("message", {}).get("content", "{}")
        try:
            data = json.loads(content)
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    async def draft(self, text: str, context: str, temperature: float = 0.0, think: bool = False) -> str:
        user = f"КОНТЕКСТ:\n{context[-3000:]}\n\nТЕКСТ:\n{text}"
        data = await self.call(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            {"type": "object", "properties": {"corrected": {"type": "string"}}, "required": ["corrected"], "additionalProperties": False},
            temperature=temperature, think=think,
        )
        corrected = data.get("corrected") if isinstance(data, dict) else None
        return corrected if isinstance(corrected, str) else text

    async def judge(self, text: str, candidates: list[EditCandidate]) -> list[bool]:
        payload = [{"id": i, "before": c.before, "after": c.after, "category": c.category} for i, c in enumerate(candidates)]
        data = await self.call(
            [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": json.dumps({"text": text, "candidates": payload}, ensure_ascii=False)}],
            {"type": "object", "properties": {"accept": {"type": "array", "items": {"type": "boolean"}, "maxItems": 16}}, "required": ["accept"], "additionalProperties": False},
        )
        flags = data.get("accept") if isinstance(data, dict) else None
        return flags if isinstance(flags, list) else []


class LocalContextEngine:
    def __init__(self) -> None:
        self.morph = None
        try:
            import pymorphy3
            self.morph = pymorphy3.MorphAnalyzer()
        except Exception as exc:
            logger.warning("LocalContextEngine: pymorphy3 unavailable: %s", exc)

    @property
    def available(self) -> bool:
        return self.morph is not None

    def candidates(self, text: str) -> list[EditCandidate]:
        return []


class LanguageToolVerifier:
    """LanguageTool is a verifier only, never the primary candidate generator."""

    def __init__(self) -> None:
        self.url = os.getenv("LANGUAGETOOL_URL", "").strip().rstrip("/")
        self.language = os.getenv("LANGUAGETOOL_LANGUAGE", "ru-RU")

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    async def _matches(self, text: str) -> list[dict[str, Any]]:
        if not self.url:
            return []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.post(f"{self.url}/v2/check", data={"language": self.language, "text": text})
                r.raise_for_status()
                data = r.json()
                return data.get("matches", []) if isinstance(data, dict) else []
        except Exception as exc:
            logger.warning("LanguageTool verifier unavailable: %s", exc)
            return []

    async def supports(self, source: str, corrected: str) -> bool:
        if not self.enabled:
            return True
        before = await self._matches(source)
        after = await self._matches(corrected)
        return len(after) <= len(before)


def diff_candidates(source: str, corrected: str, category: str, confidence: float = 0.70) -> list[EditCandidate]:
    from safe_diff import diff_candidates as bounded_diff
    return bounded_diff(source, corrected, category, confidence)


class HybridRouter:
    def __init__(self, preset: str, protected_words: set[str] | None = None) -> None:
        self.preset = preset.upper()
        if self.preset not in STACKS:
            raise RuntimeError(f"Unsupported LLM_PRESET={self.preset!r}; expected A, B, X or Y")
        self.info = STACKS[self.preset]
        self.local = LocalContextEngine()
        self.lt = LanguageToolVerifier()
        self.protected_words = protected_words or set()
        self._calls = 0

    def _model_pair(self) -> tuple[str, str]:
        if self.preset == "A":
            return TLITE, GIGACHAT
        if self.preset == "B":
            return GIGACHAT, TLITE
        return QWEN35, GIGACHAT if self.preset == "X" else TLITE

    async def _ollama_drafts(self, model: str, text: str, context: str, count: int) -> list[str]:
        client = OllamaJSON(model)
        return [await client.draft(text, context, temperature=t) for t in ([0.0, 0.25][:count])]

    async def _qwen_drafts(self, text: str, context: str, count: int) -> list[str]:
        from qwen35_backend import Qwen35Backend
        backend = Qwen35Backend()
        return [await backend.correct(text, context, temperature=t) for t in ([0.7, 0.9][:count])]

    async def candidates(self, text: str, context: str = "") -> list[EditCandidate]:
        from syntax_candidates import candidates as syntax_candidates

        local = self.local.candidates(text) + syntax_candidates(text)
        generator, judge = self._model_pair()
        self._calls += 1
        if self.preset in {"X", "Y"}:
            drafts = await self._qwen_drafts(text, context, 2 if self.preset == "Y" else 1)
        else:
            drafts = await self._ollama_drafts(generator, text, context, 1)

        generated: list[EditCandidate] = []
        for draft in drafts:
            generated.extend(diff_candidates(text, draft, "model-draft", 0.72 if len(drafts) == 1 else 0.74))
        if self.preset == "Y" and len(drafts) == 2 and drafts[0] == drafts[1]:
            generated.extend(diff_candidates(text, drafts[0], "self-consistent", 0.82))

        uniq: dict[tuple[str, str], EditCandidate] = {}
        local_keys = {(c.before, c.after) for c in local}
        for c in local + generated:
            key = (c.before, c.after)
            score = c.confidence + (0.10 if key in local_keys and not c.category.startswith("local") else 0.0)
            uniq[key] = EditCandidate(c.before, c.after, min(0.99, score), c.category, c.reason)
        merged = list(uniq.values())

        if merged:
            try:
                flags = await OllamaJSON(judge).judge(text, merged[:16])
            except Exception as exc:
                logger.warning("Cross-model judge failed: %s; keeping deterministic candidates", exc)
                flags = []
            if flags:
                merged = [c for i, c in enumerate(merged[:16]) if i < len(flags) and flags[i] is True]
            else:
                # Infrastructure failure is not a semantic rejection.
                merged = [c for c in merged if c.category.startswith("syntax-")]

        if merged and self.lt.enabled:
            corrected, _ = DecisionEngine(protected_words=self.protected_words).apply(text, merged)
            if not await self.lt.supports(text, corrected):
                merged = [c for c in merged if c.category.startswith("syntax-")]
        return merged

    async def warmup(self) -> None:
        if self.preset in {"X", "Y"}:
            from qwen35_backend import Qwen35Backend
            await Qwen35Backend().warmup()
        else:
            model, _ = self._model_pair()
            await OllamaJSON(model).draft("Проверка запуска.", "Проверка запуска.")

    def metrics(self) -> dict[str, Any]:
        return {"preset": self.preset, "calls": self._calls, "local_detector": self.local.available, "languagetool_verifier": self.lt.enabled}
