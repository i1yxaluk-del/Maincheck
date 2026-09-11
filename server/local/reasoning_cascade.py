"""Experimental two-model Russian correction cascade.

A stronger reasoning model performs the linguistic analysis. A small fast model
then extracts only the corrected text from the reasoning trace. The extracted
text still passes through the normal bounded diff, arbiter and generative guard.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from decision_engine import EditCandidate
from llm_text import sanitize
from safe_diff import diff_candidates
from segmentation import split_sentences, strip_enumeration

DEFAULT_REASONER = "deepseek-r1:7b-qwen-distill-q4_K_M"
DEFAULT_EXTRACTOR = "hf.co/loqira/Qwen3.5-0.8B-GEC-KAZ-RUS-ENG:Q4_0"

REASONER_SYSTEM = """Ты — ведущий редактор русского официально-делового текста.
Проведи подробную внутреннюю проверку орфографии, пунктуации, грамматики,
управления, согласования, логики и стилистики. Не меняй факты, числа, названия,
сокращения и смысл. Не удаляй полезную информацию. В конце обязательно напиши
маркер ИСПРАВЛЕННЫЙ ТЕКСТ: и после него полный исправленный текст без пояснений.
Размышление отключать не требуется: сначала тщательно проверь все связи."""

EXTRACTOR_SYSTEM = """Ты — точный экстрактор результата редактора.
Получишь исходный текст и полный ответ сильной размышляющей модели. Верни только
итоговый исправленный текст целиком. Не объясняй, не добавляй кавычки или маркеры.
Не исправляй текст самостоятельно и не отменяй правки сильной модели. Сохрани
переносы строк, числа, сокращения и структуру исходного текста."""


@dataclass(frozen=True)
class ReasoningCascadeStats:
    enabled: bool
    reasoner: str
    extractor: str
    calls: int
    failures: int


class ReasoningCascade:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self.enabled = os.getenv("REASONING_CASCADE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        self.url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.reasoner = os.getenv("REASONING_MODEL", DEFAULT_REASONER)
        self.extractor = os.getenv("REASONING_EXTRACTOR_MODEL", DEFAULT_EXTRACTOR)
        self.timeout = float(os.getenv("REASONING_TIMEOUT", "120"))
        self.extract_timeout = float(os.getenv("REASONING_EXTRACT_TIMEOUT", "30"))
        self.num_ctx = int(os.getenv("REASONING_NUM_CTX", "4096"))
        self.num_predict = int(os.getenv("REASONING_NUM_PREDICT", "1024"))
        self.threads = int(os.getenv("REASONING_NUM_THREADS", "16"))
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
        self.max_sentences = int(os.getenv("REASONING_MAX_SENTENCES", "6"))
        self._client = client
        self._calls = 0
        self._failures = 0

    async def _chat(self, model: str, messages: list[dict[str, str]], timeout: float,
                    num_predict: int) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0.0,
                "num_ctx": self.num_ctx,
                "num_predict": num_predict,
                "num_thread": self.threads,
                "repeat_penalty": 1.02,
            },
        }
        # Deliberately do not send think=false: the primary model keeps its
        # native reasoning mode. The extractor sees both thinking and final text.
        if self._client is not None:
            response = await self._client.post(f"{self.url}/api/chat", json=payload, timeout=timeout)
            response.raise_for_status()
            return str(response.json().get("message", {}).get("content", "")).strip()
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{self.url}/api/chat", json=payload)
            response.raise_for_status()
            return str(response.json().get("message", {}).get("content", "")).strip()

    async def correct(self, text: str, context: str = "") -> str:
        if not self.enabled or not text.strip():
            return text
        self._calls += 1
        reasoning = await self._chat(
            self.reasoner,
            [
                {"role": "system", "content": REASONER_SYSTEM},
                {"role": "user", "content": f"Контекст документа:\n{context[-1200:]}\n\nТекст:\n{text}"},
            ],
            self.timeout,
            self.num_predict,
        )
        if not reasoning:
            raise RuntimeError("reasoner returned an empty response")
        extracted = await self._chat(
            self.extractor,
            [
                {"role": "system", "content": EXTRACTOR_SYSTEM},
                {"role": "user", "content": f"ИСХОДНЫЙ ТЕКСТ:\n{text}\n\nОТВЕТ СИЛЬНОЙ МОДЕЛИ:\n{reasoning}"},
            ],
            self.extract_timeout,
            min(512, self.num_predict),
        )
        cleaned = sanitize(text, extracted)
        if not cleaned:
            raise RuntimeError("extractor did not return a safe local correction")
        return cleaned

    async def candidates(self, text: str, context: str = "") -> list[EditCandidate]:
        if not self.enabled:
            return []
        out: list[EditCandidate] = []
        for segment in [strip_enumeration(s) for s in split_sentences(text)][:self.max_sentences]:
            try:
                corrected = await self.correct(segment.text, context)
            except Exception:
                self._failures += 1
                continue
            out.extend(diff_candidates(
                segment.text, corrected, "russian-gec-reasoning", 0.86,
                offset=segment.start,
            ))
        return out

    def metrics(self) -> ReasoningCascadeStats:
        return ReasoningCascadeStats(
            self.enabled, self.reasoner, self.extractor, self._calls, self._failures,
        )
