"""Локальный GEC-специалист поверх Ollama (v9).

Изменения относительно v8
=========================
* **Few-shot из банка эталонов.** В v8 `RetrievalExamples` подключался
  только к rescue-моделям (T-lite/GigaChat), а специалист на 0.8 B
  параметров — самая нуждающаяся в подсказках модель — работал без
  примеров. Теперь 2135 пар «неверно → верно» из
  `shared/gec_seed/*.jsonl` доступны и ему.
* **Работа по предложениям.** Вызывающая сторона (`hybrid_editor`)
  передаёт по одному предложению: на квантованной Q4_0-модели такого
  размера вероятность потери фрагмента при переписывании абзаца
  неприемлемо высока.
* **Общий HTTP-клиент** вместо `AsyncClient` на каждый вызов.
* Ответ модели нормализуется в `llm_text.sanitize` на стороне
  вызывающего кода: обёртки «Исправленный текст:», ``` и `<think>`
  раньше превращали diff в глобальный и стадия молча давала ноль правок.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx

logger = logging.getLogger("ai_suggester.ollama_gec")

DEFAULT_MODEL = "hf.co/loqira/Qwen3.5-0.8B-GEC-KAZ-RUS-ENG:Q4_0"
SYSTEM_PROMPT = (
    "Ты — специализированный корректор русского официально-делового текста. "
    "Исправляй только объективные ошибки орфографии, пунктуации, грамматики, "
    "согласования, управления и синтаксиса. Отдельно проверяй падеж и число "
    "существительных после числительных, местоимений и количественных слов. "
    "Сохраняй язык, все исходные слова и смысл, порядок слов, переносы строк и структуру. "
    "Не перефразируй, не сокращай, не добавляй текст и не разбивай предложение на новые абзацы. "
    "Верни только исправленный исходный текст целиком, без пояснений, "
    "без кавычек и без вводных слов."
)


@dataclass(frozen=True)
class OllamaGecStats:
    enabled: bool
    model: str
    calls: int
    few_shot: int = 0


class OllamaGecSpecialist:
    """Русский GEC-специалист, работающий в уже запущенном демоне Ollama."""

    def __init__(self, retriever=None, client: httpx.AsyncClient | None = None) -> None:
        self.enabled = os.getenv("OLLAMA_GEC_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        self.model = os.getenv("OLLAMA_GEC_MODEL", DEFAULT_MODEL)
        self.url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.timeout = float(os.getenv("OLLAMA_GEC_TIMEOUT", "30"))
        self.num_ctx = int(os.getenv("OLLAMA_GEC_NUM_CTX", "2048"))
        self.num_predict = int(os.getenv("OLLAMA_GEC_NUM_PREDICT", "256"))
        self.num_thread = int(os.getenv("OLLAMA_GEC_NUM_THREAD", os.getenv("NUM_THREADS", "16")))
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
        self.few_shot = int(os.getenv("OLLAMA_GEC_FEW_SHOT", "3"))
        self.retriever = retriever
        self._client = client
        self._calls = 0

    @property
    def available(self) -> bool:
        return self.enabled

    def _messages(self, text: str) -> list[dict[str, str]]:
        """Собирает диалог с few-shot примерами из банка эталонов.

        Примеры подаются как полноценные пары ролей user/assistant, а не
        как текстовый список в системном промпте: маленькие
        инструктивные модели существенно точнее следуют формату, когда
        видят его в истории диалога.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if self.retriever is not None and self.few_shot > 0:
            try:
                pairs = self.retriever.pairs(text, top_k=self.few_shot)
            except Exception as exc:  # pragma: no cover
                logger.warning("GEC few-shot unavailable: %s", exc)
                pairs = []
            for wrong, right, _rule in pairs[: self.few_shot]:
                messages.append({"role": "user", "content": wrong})
                messages.append({"role": "assistant", "content": right})
        messages.append({"role": "user", "content": text})
        return messages

    async def correct(self, text: str) -> str:
        if not self.enabled or not text.strip():
            return text
        self._calls += 1
        payload = {
            "model": self.model,
            "messages": self._messages(text),
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
        if self._client is not None:
            response = await self._client.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        else:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        return str(data.get("message", {}).get("content", "")).strip()

    async def warmup(self) -> None:
        if self.enabled:
            await self.correct("Проверка запуска корректора.")

    def metrics(self) -> OllamaGecStats:
        return OllamaGecStats(self.enabled, self.model, self._calls, self.few_shot)
