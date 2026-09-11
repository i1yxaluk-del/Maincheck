"""Russian comma restoration stage based on RUPunct.

Unlike a generative LLM, the model classifies punctuation after each word.  We
only consume the COMMA class and project it onto the original text; spelling,
capitalization, numbers, quotes and line layout are never regenerated.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass

from decision_engine import EditCandidate
from punctuation_pipeline import collapse_soft_breaks

logger = logging.getLogger("ai_suggester.rupunct")
WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z]+(?:-[А-Яа-яЁёA-Za-z]+)*|\d+")


@dataclass(frozen=True)
class RuPunctStats:
    enabled: bool
    loaded: bool
    model: str
    calls: int
    candidates: int
    failures: int


class RuPunctStage:
    """Fast token-classification source for missing and redundant commas."""

    def __init__(self) -> None:
        self.enabled = os.getenv("RUPUNCT_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        self.model_id = os.getenv("RUPUNCT_MODEL", "RUPunct/RUPunct_big")
        self.threads = int(os.getenv("RUPUNCT_THREADS", "8"))
        self.max_words = int(os.getenv("RUPUNCT_MAX_WORDS", "220"))
        self.add_threshold = float(os.getenv("RUPUNCT_ADD_THRESHOLD", "0.88"))
        self.remove_threshold = float(os.getenv("RUPUNCT_REMOVE_THRESHOLD", "0.96"))
        self._classifier = None
        self._lock = asyncio.Lock()
        self._calls = 0
        self._candidates = 0
        self._failures = 0

    @property
    def available(self) -> bool:
        return self.enabled

    def _load(self) -> None:
        if self._classifier is not None:
            return
        import torch
        from transformers import AutoTokenizer, pipeline
        try:
            torch.set_num_threads(max(1, self.threads))
        except RuntimeError:
            pass
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, strip_accents=False, add_prefix_space=True,
        )
        self._classifier = pipeline(
            "ner", model=self.model_id, tokenizer=tokenizer,
            aggregation_strategy="first", device=-1,
        )
        logger.info("Loaded Russian punctuation classifier: %s", self.model_id)

    async def candidates(self, text: str) -> list[EditCandidate]:
        if not self.enabled or not text.strip():
            return []
        self._calls += 1
        async with self._lock:
            try:
                result = await asyncio.to_thread(self._candidates_sync, text)
            except Exception as exc:
                self._failures += 1
                logger.warning("RuPunct failed: %s", exc)
                return []
        self._candidates += len(result)
        return result

    def _candidates_sync(self, text: str) -> list[EditCandidate]:
        self._load()
        flat = collapse_soft_breaks(text)
        words = list(WORD_RE.finditer(flat))
        if len(words) < 3 or len(words) > self.max_words:
            return []
        plain = " ".join(match.group(0) for match in words)
        predictions = list(self._classifier(plain))
        # The official model card uses aggregation_strategy=first and emits one
        # label per whitespace-delimited word.  Any mismatch is unsafe.
        if len(predictions) != len(words):
            logger.debug("RuPunct alignment mismatch: words=%d predictions=%d", len(words), len(predictions))
            return []

        # Positions are needed in the original text, not in the flattened copy.
        original_words = list(WORD_RE.finditer(text))
        if len(original_words) != len(words):
            return []
        out: list[EditCandidate] = []
        for index, (word, prediction) in enumerate(zip(original_words, predictions)):
            if index + 1 >= len(original_words):
                break
            label = str(prediction.get("entity_group", "")).upper()
            score = float(prediction.get("score", 0.0))
            gap_start = word.end()
            gap_end = original_words[index + 1].start()
            gap = text[gap_start:gap_end]
            comma_match = re.search(r",", gap)
            predicts_comma = label.endswith("_COMMA")
            if predicts_comma and comma_match is None and score >= self.add_threshold:
                before = word.group(0)
                out.append(EditCandidate(
                    before, before + ",", score, "sage-rupunct",
                    "RuPunct: высокая вероятность запятой на границе слов",
                    start=word.start(),
                ))
            elif not predicts_comma and comma_match is not None and score >= self.remove_threshold:
                comma_pos = gap_start + comma_match.start()
                out.append(EditCandidate(
                    ",", "", score, "sage-rupunct",
                    "RuPunct: высокая вероятность лишней запятой",
                    start=comma_pos,
                ))
        return out

    async def warmup(self) -> None:
        if self.enabled:
            await self.candidates("Проверка выполнена, результаты оформлены.")

    def metrics(self) -> RuPunctStats:
        return RuPunctStats(
            self.enabled, self._classifier is not None, self.model_id,
            self._calls, self._candidates, self._failures,
        )
