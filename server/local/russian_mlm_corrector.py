from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass

logger = logging.getLogger("ai_suggester.russian_mlm")

DEFAULT_MODEL = "ai-forever/ruBert-base"
WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z-]{2,}")


@dataclass(frozen=True)
class MlmStats:
    enabled: bool
    loaded: bool
    model: str
    calls: int
    positions: int


class RussianMlmCorrector:
    """Fast Russian MLM candidate generator.

    It does not rewrite the document. It masks one whole-token word at a time,
    proposes high-probability alternatives from a Russian pretrained encoder,
    and returns only minimal word substitutions. This keeps recall high without
    allowing a generative model to reorder or paraphrase the sentence.
    """

    def __init__(self) -> None:
        self.enabled = os.getenv("RUSSIAN_MLM_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        self.model_id = os.getenv("RUSSIAN_MLM_MODEL", DEFAULT_MODEL)
        self.top_k = int(os.getenv("RUSSIAN_MLM_TOP_K", "8"))
        self.max_positions = int(os.getenv("RUSSIAN_MLM_MAX_POSITIONS", "48"))
        self.batch_size = int(os.getenv("RUSSIAN_MLM_BATCH", "12"))
        self.min_margin = float(os.getenv("RUSSIAN_MLM_MIN_MARGIN", "0.15"))
        self._tokenizer = None
        self._model = None
        self._calls = 0
        self._positions = 0
        self._lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        return self.enabled

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        logger.info("Loading Russian MLM candidate generator: %s", self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForMaskedLM.from_pretrained(self.model_id, torch_dtype=torch.float32)
        self._model.eval()
        try:
            torch.set_num_threads(max(1, int(os.getenv("RUSSIAN_MLM_THREADS", "4"))))
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    @staticmethod
    def _is_word_token(token: str) -> bool:
        return bool(token) and bool(re.fullmatch(r"[А-Яа-яЁёA-Za-z-]+", token))

    @staticmethod
    def _normalize_candidate(candidate: str, source: str) -> str:
        if source[:1].isupper():
            return candidate[:1].upper() + candidate[1:]
        return candidate.lower()

    def _mask_positions(self, text: str):
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        offsets = inputs.pop("offset_mapping")[0].tolist()
        input_ids = inputs["input_ids"][0].tolist()
        tokens = self._tokenizer.convert_ids_to_tokens(input_ids)
        mask_id = self._tokenizer.mask_token_id
        positions = []
        for m in WORD_RE.finditer(text):
            overlaps = [
                i for i, (start, end) in enumerate(offsets)
                if start < m.end() and end > m.start() and end > start
            ]
            if len(overlaps) != 1:
                continue
            i = overlaps[0]
            if i <= 0 or i >= len(tokens) - 1:
                continue
            if not self._is_word_token(tokens[i].replace("##", "")):
                continue
            positions.append((i, m.start(), m.end(), m.group(0), input_ids[i]))
            if len(positions) >= self.max_positions:
                break
        return inputs, positions, mask_id

    def _correct_sync(self, text: str):
        import torch
        from decision_engine import EditCandidate

        self._load()
        inputs, positions, mask_id = self._mask_positions(text)
        if not positions or mask_id is None:
            return []

        self._positions += len(positions)
        out = []
        base_ids = inputs["input_ids"][0].tolist()
        id_to_word = {idx: tok for idx, tok in enumerate(self._tokenizer.get_vocab())}
        for start in range(0, len(positions), self.batch_size):
            batch = positions[start : start + self.batch_size]
            ids_batch = torch.tensor([base_ids[:] for _ in batch], dtype=torch.long)
            attention = inputs["attention_mask"].repeat(len(batch), 1)
            for row, (pos, *_rest) in enumerate(batch):
                ids_batch[row, pos] = mask_id
            with torch.inference_mode():
                logits = self._model(input_ids=ids_batch, attention_mask=attention).logits
            for row, (pos, _start, _end, source_word, original_id) in enumerate(batch):
                scores = logits[row, pos]
                top_scores, top_ids = torch.topk(scores, k=min(self.top_k + 4, scores.shape[-1]))
                original_score = float(scores[original_id])
                alternatives = []
                for score, token_id in zip(top_scores.tolist(), top_ids.tolist()):
                    if token_id == original_id:
                        continue
                    token = id_to_word.get(token_id, "")
                    token = token.replace("##", "")
                    if not self._is_word_token(token):
                        continue
                    candidate = self._normalize_candidate(token, source_word)
                    if candidate.lower() == source_word.lower():
                        continue
                    margin = float(score) - original_score
                    if margin < self.min_margin:
                        continue
                    alternatives.append((margin, candidate))
                if not alternatives:
                    continue
                alternatives.sort(reverse=True)
                margin, candidate = alternatives[0]
                out.append(EditCandidate(
                    source_word,
                    candidate,
                    min(0.93, 0.68 + max(0.0, margin) * 0.08),
                    "russian-mlm",
                    f"ruBert masked-token preference margin={margin:.2f}",
                ))
        return out

    async def candidates(self, text: str):
        if not self.enabled or not text.strip():
            return []
        self._calls += 1
        async with self._lock:
            return await asyncio.to_thread(self._correct_sync, text)

    async def warmup(self) -> None:
        if self.enabled:
            await self.candidates("Проверка запуска корректора.")

    def metrics(self) -> MlmStats:
        return MlmStats(self.enabled, self._model is not None, self.model_id, self._calls, self._positions)
