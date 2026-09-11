"""SAGE-корректор орфографии и пунктуации (v9).

Изменения относительно v8
=========================
* **Пакетная генерация.** Предложения фрагмента прогоняются одним
  вызовом `generate` с padding-ом. Attention квадратично по длине, и
  батч из пяти коротких предложений считается быстрее одного длинного
  абзаца при лучшем качестве: модель `sage-fredt5-distilled-95m`
  обучалась на одиночных предложениях.
* **Явный `GenerationConfig`.** Убирает конфликт `max_new_tokens` и
  `max_length`, из-за которого transformers на каждый запрос писал в
  журнал предупреждение, а генерация могла обрезаться по 256 токенам
  вместе с промптом.
* **Динамическая int8-квантизация (опционально).** На Xeon E5-2690 v4
  (Broadwell, AVX2, без AVX-512 и без bf16) `quantize_dynamic` по
  `nn.Linear` даёт кратное ускорение seq2seq на CPU при неизменном
  выводе на коротких предложениях. Включается `SAGE_QUANTIZE=int8`.
* **Настраиваемое число потоков.** Значение по умолчанию поднято с 2 до
  8: на 32-ядерном сервере два потока были узким местом (1.2–2.7 с на
  фрагмент в 200 символов по журналам).
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger("ai_suggester.quality_models")
DEFAULT_SPELL_MODEL = "ai-forever/sage-fredt5-distilled-95m"


@dataclass(frozen=True)
class QualityModelStats:
    enabled: bool
    loaded: bool
    model: str
    calls: int
    quantized: bool = False


class SageRussianCorrector:
    """Быстрый генератор кандидатов по орфографии и пунктуации."""

    def __init__(self) -> None:
        self.enabled = os.getenv("SAGE_CORRECTOR_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        self.model_id = os.getenv("SAGE_CORRECTOR_MODEL", DEFAULT_SPELL_MODEL)
        self.max_new_tokens = int(os.getenv("SAGE_CORRECTOR_MAX_NEW_TOKENS", "192"))
        self.max_input_tokens = int(os.getenv("SAGE_CORRECTOR_MAX_INPUT_TOKENS", "384"))
        self.num_beams = int(os.getenv("SAGE_CORRECTOR_NUM_BEAMS", "1"))
        self.threads = int(os.getenv("SAGE_CORRECTOR_THREADS", "8"))
        self.quantize = os.getenv("SAGE_QUANTIZE", "none").strip().lower()
        self.max_batch = int(os.getenv("SAGE_CORRECTOR_MAX_BATCH", "8"))
        self._tokenizer = None
        self._model = None
        self._generation_config = None
        self._quantized = False
        self._calls = 0
        self._lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        return self.enabled

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

        try:
            torch.set_num_threads(max(1, self.threads))
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        logger.info("Loading Russian spelling/punctuation specialist: %s", self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, dtype=torch.float32)
        model.eval()

        if self.quantize == "int8":
            try:
                model = torch.ao.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8,
                )
                self._quantized = True
                logger.info("SAGE: включена динамическая int8-квантизация")
            except Exception as exc:
                logger.warning("SAGE: int8-квантизация недоступна (%s), работаем в fp32", exc)

        self._model = model
        # Явная конфигурация генерации: не смешиваем max_length и
        # max_new_tokens, иначе transformers обрезает вывод по 256
        # токенам вместе с промптом и пишет предупреждение на каждый вызов.
        self._generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            num_beams=max(1, self.num_beams),
            do_sample=False,
            no_repeat_ngram_size=3,
            early_stopping=self.num_beams > 1,
        )

    async def correct(self, text: str) -> str:
        results = await self.correct_batch([text])
        return results[0] if results else text

    async def correct_batch(self, texts: list[str]) -> list[str]:
        """Исправляет список предложений одним прогоном модели."""
        if not self.enabled:
            return list(texts)
        payload = [t for t in texts if t and t.strip()]
        if not payload:
            return list(texts)
        self._calls += 1
        async with self._lock:
            corrected = await asyncio.to_thread(self._correct_sync, payload)
        iterator = iter(corrected)
        return [next(iterator) if (t and t.strip()) else t for t in texts]

    def _correct_sync(self, texts: list[str]) -> list[str]:
        import torch

        self._load()
        out: list[str] = []
        for start in range(0, len(texts), max(1, self.max_batch)):
            chunk = [t.strip() for t in texts[start:start + max(1, self.max_batch)]]
            inputs = self._tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_input_tokens,
            )
            with torch.inference_mode():
                output = self._model.generate(**inputs, generation_config=self._generation_config)
            out.extend(
                text.strip()
                for text in self._tokenizer.batch_decode(output, skip_special_tokens=True)
            )
        return out

    async def warmup(self) -> None:
        if self.enabled:
            await self.correct_batch(["Проверка запуска корректора."])

    def metrics(self) -> QualityModelStats:
        return QualityModelStats(
            self.enabled, self._model is not None, self.model_id, self._calls, self._quantized,
        )
