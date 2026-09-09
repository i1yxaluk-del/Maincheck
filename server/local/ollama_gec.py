from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


DEFAULT_MODEL = "qwen3.5-gec"
SYSTEM_PROMPT = (
    "Отформатируй текст голосового ввода: расставь пунктуацию и заглавные буквы, "
    "разбей на абзацы, исправь опечатки и орфографические ошибки. Сохрани язык, "
    "слова и смысл, ничего не добавляй от себя."
)


@dataclass(frozen=True)
class OllamaGecStats:
    enabled: bool
    model: str
    calls: int


class OllamaGecSpecialist:
    """Fast local GEC/proofreading specialist running inside the existing Ollama."""

    def __init__(self) -> None:
        self.enabled = os.getenv("OLLAMA_GEC_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        self.model = os.getenv("OLLAMA_GEC_MODEL", DEFAULT_MODEL)
        self.url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.timeout = float(os.getenv("OLLAMA_GEC_TIMEOUT", "30"))
        self.num_ctx = int(os.getenv("OLLAMA_GEC_NUM_CTX", "2048"))
        self.num_predict = int(os.getenv("OLLAMA_GEC_NUM_PREDICT", "256"))
        self.num_thread = int(os.getenv("OLLAMA_GEC_NUM_THREAD", os.getenv("NUM_THREADS", "16")))
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
        self._calls = 0

    @property
    def available(self) -> bool:
        return self.enabled

    async def correct(self, text: str) -> str:
        if not self.enabled or not text.strip():
            return text
        self._calls += 1
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0.0,
                "top_k": 1,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "num_thread": self.num_thread,
                "repeat_penalty": 1.0,
            },
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        return str(data.get("message", {}).get("content", "")).strip()

    async def warmup(self) -> None:
        if self.enabled:
            await self.correct("Проверка запуска корректора.")

    def metrics(self) -> OllamaGecStats:
        return OllamaGecStats(self.enabled, self.model, self._calls)
