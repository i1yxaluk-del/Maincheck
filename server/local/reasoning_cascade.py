"""Optional reasoning fallback for difficult Russian corrections."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import httpx

from decision_engine import EditCandidate
from llm_text import sanitize
from safe_diff import diff_candidates
from segmentation import split_sentences, strip_enumeration

DEFAULT_REASONER = "deepseek-r1:7b-qwen-distill-q4_K_M"
DEFAULT_EXTRACTOR = "hf.co/loqira/Qwen3.5-0.8B-GEC-KAZ-RUS-ENG:Q4_0"
FINAL_RE = re.compile(r"(?:ИСПРАВЛЕННЫЙ\s+ТЕКСТ|ИТОГОВЫЙ\s+ТЕКСТ)\s*:\s*(.+)$", re.I | re.S)

REASONER_SYSTEM = """Ты — ведущий редактор русского официально-делового текста.
Проверь орфографию, пунктуацию, грамматику, управление, согласование и логику.
Не меняй факты, числа, названия, сокращения и смысл. В конце обязательно напиши
ИСПРАВЛЕННЫЙ ТЕКСТ: и полный исправленный текст без последующих пояснений."""
EXTRACTOR_SYSTEM = """Извлеки из ответа редактора только итоговый исправленный
текст. Не исправляй самостоятельно, не добавляй пояснения, кавычки или маркеры."""


@dataclass(frozen=True)
class ReasoningCascadeStats:
    enabled: bool
    reasoner: str
    extractor: str
    calls: int
    failures: int
    direct_results: int
    extractor_calls: int


class ReasoningCascade:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self.enabled = os.getenv("REASONING_CASCADE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        self.url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.reasoner = os.getenv("REASONING_MODEL", DEFAULT_REASONER)
        self.extractor = os.getenv("REASONING_EXTRACTOR_MODEL", DEFAULT_EXTRACTOR)
        self.timeout = float(os.getenv("REASONING_TIMEOUT", "60"))
        self.extract_timeout = float(os.getenv("REASONING_EXTRACT_TIMEOUT", "25"))
        self.num_ctx = int(os.getenv("REASONING_NUM_CTX", "3072"))
        self.num_predict = int(os.getenv("REASONING_NUM_PREDICT", "512"))
        self.threads = int(os.getenv("REASONING_NUM_THREADS", "16"))
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
        self.max_sentences = int(os.getenv("REASONING_MAX_SENTENCES", "4"))
        self._client = client
        self._calls = 0
        self._failures = 0
        self._direct_results = 0
        self._extractor_calls = 0

    async def _chat(self, model: str, messages: list[dict[str, str]], timeout: float,
                    num_predict: int) -> str:
        payload = {
            "model": model, "messages": messages, "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.0, "num_ctx": self.num_ctx,
                        "num_predict": num_predict, "num_thread": self.threads,
                        "repeat_penalty": 1.02},
        }
        if self._client is not None:
            response = await self._client.post(f"{self.url}/api/chat", json=payload, timeout=timeout)
            response.raise_for_status()
            message = response.json().get("message", {})
        else:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{self.url}/api/chat", json=payload)
                response.raise_for_status()
                message = response.json().get("message", {})
        thinking = str(message.get("thinking", "")).strip()
        content = str(message.get("content", "")).strip()
        return (thinking + "\n" + content).strip() if thinking else content

    async def correct(self, text: str, context: str = "") -> str:
        if not self.enabled or not text.strip():
            return text
        self._calls += 1
        reasoning = await self._chat(
            self.reasoner,
            [{"role": "system", "content": REASONER_SYSTEM},
             {"role": "user", "content": f"Контекст:\n{context[-800:]}\n\nТекст:\n{text}"}],
            self.timeout, self.num_predict,
        )
        if not reasoning:
            raise RuntimeError("reasoner returned an empty response")
        marked = FINAL_RE.search(reasoning)
        if marked:
            cleaned = sanitize(text, marked.group(1).strip())
            if cleaned:
                self._direct_results += 1
                return cleaned
        self._extractor_calls += 1
        extracted = await self._chat(
            self.extractor,
            [{"role": "system", "content": EXTRACTOR_SYSTEM},
             {"role": "user", "content": f"ИСХОДНЫЙ ТЕКСТ:\n{text}\n\nОТВЕТ РЕДАКТОРА:\n{reasoning}"}],
            self.extract_timeout, min(384, self.num_predict),
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
            self.enabled, self.reasoner, self.extractor, self._calls,
            self._failures, self._direct_results, self._extractor_calls,
        )
