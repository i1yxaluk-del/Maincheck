from __future__ import annotations

import asyncio
import os

import httpx


class Qwen35Backend:
    """Compatibility wrapper for the v3.1 X/Y text-correction backend.

    X/Y no longer load a multi-gigabyte Transformers checkpoint in the FastAPI
    process. They call the same Ollama server already used by A/B, using a
    compact Russian GEC GGUF. This keeps the single-server architecture and
    avoids PIL/torchvision and CPU/disk offload startup storms.
    """

    MODEL_ID = os.getenv(
        "GEC_SPECIALIST_MODEL",
        "hf.co/loqira/Qwen3.5-0.8B-GEC-KAZ-RUS-ENG:Q4_0",
    )
    _shared_instance: "Qwen35Backend | None" = None

    SYSTEM_PROMPT = (
        "Отформатируй текст голосового ввода: расставь пунктуацию и заглавные буквы, "
        "разбей на абзацы, исправь опечатки и орфографические ошибки. Сохрани язык, "
        "слова и смысл, ничего не добавляй от себя."
    )

    def __new__(cls):
        if cls._shared_instance is None:
            cls._shared_instance = super().__new__(cls)
            cls._shared_instance._initialized = False
        return cls._shared_instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self.url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.timeout = float(os.getenv("GEC_SPECIALIST_TIMEOUT", "180"))
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
        self.num_ctx = int(os.getenv("GEC_SPECIALIST_NUM_CTX", "2048"))
        self.max_new = int(os.getenv("GEC_SPECIALIST_MAX_NEW_TOKENS", "512"))
        self._initialized = True

    async def correct(self, text: str, context: str, temperature: float = 0.10) -> str:
        return await asyncio.to_thread(self._correct_sync, text, context, temperature)

    def _correct_sync(self, text: str, context: str, temperature: float) -> str:
        # The model is trained as a single-turn formatter. Keep the request
        # deliberately compact; context is only a small optional hint.
        context_hint = context[-1200:].strip() if context else ""
        user = text
        if context_hint:
            user = f"Контекст:\n{context_hint}\n\nТекст для исправления:\n{text}"
        payload = {
            "model": self.MODEL_ID,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0.0 if temperature <= 0 else min(float(temperature), 0.2),
                "top_k": 1,
                "num_ctx": self.num_ctx,
                "num_predict": self.max_new,
                "repeat_penalty": 1.0,
            },
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        content = data.get("message", {}).get("content", "")
        return str(content).strip()

    async def warmup(self) -> None:
        await self.correct("Проверка запуска.", "", temperature=0.0)
