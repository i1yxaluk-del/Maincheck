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


class SageRussianCorrector:
    """Fast dedicated Russian spelling/punctuation candidate generator.

    This is deliberately a candidate source, not an unconditional rewriter. The
    service converts its corrected draft into bounded edit candidates before the
    normal DecisionEngine applies anything.
    """

    def __init__(self) -> None:
        self.enabled = os.getenv("SAGE_CORRECTOR_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        self.model_id = os.getenv("SAGE_CORRECTOR_MODEL", DEFAULT_SPELL_MODEL)
        self.max_new_tokens = int(os.getenv("SAGE_CORRECTOR_MAX_NEW_TOKENS", "256"))
        self.num_beams = int(os.getenv("SAGE_CORRECTOR_NUM_BEAMS", "2"))
        self._tokenizer = None
        self._model = None
        self._calls = 0
        self._lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        return self.enabled

    async def correct(self, text: str) -> str:
        if not self.enabled:
            return text
        self._calls += 1
        async with self._lock:
            return await asyncio.to_thread(self._correct_sync, text)

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("Loading Russian spelling/punctuation model: %s", self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
        )
        self._model.eval()

    def _correct_sync(self, text: str) -> str:
        import torch

        self._load()
        prompt = text.strip()
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=max(1, self.num_beams),
                do_sample=False,
                no_repeat_ngram_size=3,
            )
        return self._tokenizer.decode(output[0], skip_special_tokens=True).strip()

    async def warmup(self) -> None:
        if self.enabled:
            await self.correct("Проверка запуска корректора.")

    def metrics(self) -> QualityModelStats:
        return QualityModelStats(self.enabled, self._model is not None, self.model_id, self._calls)
