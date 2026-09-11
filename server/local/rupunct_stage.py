"""Russian comma restoration stage based on RUPunct."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
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
    alignment_mismatches: int
    low_confidence: int
    total_ms: int


class RuPunctStage:
    def __init__(self) -> None:
        self.enabled = os.getenv("RUPUNCT_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        self.model_id = os.getenv("RUPUNCT_MODEL", "RUPunct/RUPunct_big")
        self.threads = int(os.getenv("RUPUNCT_THREADS", "8"))
        self.max_words = int(os.getenv("RUPUNCT_MAX_WORDS", "220"))
        self.add_threshold = float(os.getenv("RUPUNCT_ADD_THRESHOLD", "0.88"))
        self.remove_threshold = float(os.getenv("RUPUNCT_REMOVE_THRESHOLD", "0.96"))
        self._classifier = None
        self._lock = asyncio.Lock()
        self._calls = self._candidates = self._failures = 0
        self._alignment_mismatches = self._low_confidence = self._total_ms = 0

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
        started = time.perf_counter()
        async with self._lock:
            try:
                result = await asyncio.to_thread(self._candidates_sync, text)
            except Exception as exc:
                self._failures += 1
                logger.warning("RuPunct failed: %s", exc)
                return []
            finally:
                self._total_ms += int((time.perf_counter() - started) * 1000)
        self._candidates += len(result)
        return result

    @staticmethod
    def _plain_with_spans(words) -> tuple[str, list[tuple[int, int]]]:
        chunks: list[str] = []
        spans: list[tuple[int, int]] = []
        cursor = 0
        for index, word in enumerate(words):
            if index:
                chunks.append(" ")
                cursor += 1
            token = word.group(0)
            spans.append((cursor, cursor + len(token)))
            chunks.append(token)
            cursor += len(token)
        return "".join(chunks), spans

    def _align(self, predictions, spans):
        if predictions and all("start" in p and "end" in p for p in predictions):
            aligned = []
            for start, end in spans:
                overlapping = [p for p in predictions
                               if int(p["end"]) > start and int(p["start"]) < end]
                if not overlapping:
                    return None
                # A hyphenated word can produce several aggregated pieces; the
                # final piece owns punctuation at the outer word boundary.
                aligned.append(max(overlapping, key=lambda p: int(p["end"])))
            return aligned
        return list(predictions) if len(predictions) == len(spans) else None

    def _candidates_sync(self, text: str) -> list[EditCandidate]:
        self._load()
        flat = collapse_soft_breaks(text)
        flat_words = list(WORD_RE.finditer(flat))
        if len(flat_words) < 3 or len(flat_words) > self.max_words:
            return []
        plain, spans = self._plain_with_spans(flat_words)
        predictions = list(self._classifier(plain))
        aligned = self._align(predictions, spans)
        if aligned is None:
            self._alignment_mismatches += 1
            logger.info("RuPunct alignment mismatch: words=%d predictions=%d", len(flat_words), len(predictions))
            return []

        original_words = list(WORD_RE.finditer(text))
        if len(original_words) != len(flat_words):
            self._alignment_mismatches += 1
            return []
        out: list[EditCandidate] = []
        for index, (word, prediction) in enumerate(zip(original_words, aligned)):
            if index + 1 >= len(original_words):
                break
            label = str(prediction.get("entity_group", "")).upper()
            score = float(prediction.get("score", 0.0))
            gap_start, gap_end = word.end(), original_words[index + 1].start()
            gap = text[gap_start:gap_end]
            comma_match = re.search(r",", gap)
            predicts_comma = label.endswith("_COMMA")
            threshold = self.add_threshold if predicts_comma else self.remove_threshold
            if score < threshold:
                if predicts_comma or comma_match is not None:
                    self._low_confidence += 1
                continue
            if predicts_comma and comma_match is None:
                before = word.group(0)
                out.append(EditCandidate(
                    before, before + ",", score, "sage-rupunct",
                    "RuPunct: высокая вероятность запятой на границе слов",
                    start=word.start(),
                ))
            elif not predicts_comma and comma_match is not None:
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
            self._alignment_mismatches, self._low_confidence, self._total_ms,
        )
