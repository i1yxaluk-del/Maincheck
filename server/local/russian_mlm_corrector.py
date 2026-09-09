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
    """Russian masked-LM candidate detector; never rewrites the document."""

    def __init__(self) -> None:
        self.enabled = os.getenv("RUSSIAN_MLM_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        self.model_id = os.getenv("RUSSIAN_MLM_MODEL", DEFAULT_MODEL)
        self.top_k = int(os.getenv("RUSSIAN_MLM_TOP_K", "12"))
        self.max_positions = int(os.getenv("RUSSIAN_MLM_MAX_POSITIONS", "40"))
        self.batch_size = int(os.getenv("RUSSIAN_MLM_BATCH", "8"))
        self.min_margin = float(os.getenv("RUSSIAN_MLM_MIN_MARGIN", "0.10"))
        self._tokenizer = None
        self._model = None
        self._morph = None
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
        self._model = AutoModelForMaskedLM.from_pretrained(self.model_id, dtype=torch.float32)
        self._model.eval()
        try:
            torch.set_num_threads(max(1, int(os.getenv("RUSSIAN_MLM_THREADS", "4"))))
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    def _morphology_ok(self, source: str, candidate: str) -> bool:
        if self._morph is None:
            try:
                import pymorphy3
                self._morph = pymorphy3.MorphAnalyzer()
            except Exception:
                self._morph = False
        if self._morph:
            source_norms = {p.normal_form for p in self._morph.parse(source.lower())[:3] if p.is_known}
            candidate_norms = {p.normal_form for p in self._morph.parse(candidate.lower())[:3] if p.is_known}
            if source_norms and candidate_norms and source_norms & candidate_norms:
                return True
        a, b = source.lower(), candidate.lower()
        if abs(len(a) - len(b)) > max(2, int(len(a) * 0.35)):
            return False
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1] <= max(3, int(len(a) * 0.30))

    @staticmethod
    def _is_word_token(token: str) -> bool:
        return bool(token) and bool(re.fullmatch(r"[А-Яа-яЁёA-Za-z-]+", token))

    @staticmethod
    def _normalize(candidate: str, source: str) -> str:
        candidate = candidate.replace("##", "")
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
        )
        offsets = inputs.pop("offset_mapping")[0].tolist()
        ids = inputs["input_ids"][0].tolist()
        tokens = self._tokenizer.convert_ids_to_tokens(ids)
        positions = []
        for m in WORD_RE.finditer(text):
            overlaps = [
                i for i, (start, end) in enumerate(offsets)
                if start < m.end() and end > m.start() and end > start
            ]
            if len(overlaps) != 1:
                continue
            pos = overlaps[0]
            if pos <= 0 or pos >= len(tokens) - 1:
                continue
            if not self._is_word_token(tokens[pos]):
                continue
            positions.append((pos, m.group(0), ids[pos]))
            if len(positions) >= self.max_positions:
                break
        return inputs, positions, self._tokenizer.mask_token_id

    def _correct_sync(self, text: str):
        import torch
        from decision_engine import EditCandidate

        self._load()
        inputs, positions, mask_id = self._mask_positions(text)
        if not positions or mask_id is None:
            return []
        self._positions += len(positions)
        base_ids = inputs["input_ids"][0].tolist()
        id_to_token = {idx: token for token, idx in self._tokenizer.get_vocab().items()}
        out = []

        for start in range(0, len(positions), self.batch_size):
            batch = positions[start : start + self.batch_size]
            batch_ids = torch.tensor([base_ids[:] for _ in batch], dtype=torch.long)
            attention = inputs["attention_mask"].repeat(len(batch), 1)
            for row, (pos, _source, _original_id) in enumerate(batch):
                batch_ids[row, pos] = mask_id
            with torch.inference_mode():
                logits = self._model(input_ids=batch_ids, attention_mask=attention).logits

            for row, (pos, source_word, original_id) in enumerate(batch):
                scores = logits[row, pos]
                top_scores, top_ids = torch.topk(scores, k=min(self.top_k + 6, scores.shape[-1]))
                original_score = float(scores[original_id])
                best = None
                for score, token_id in zip(top_scores.tolist(), top_ids.tolist()):
                    if token_id == original_id:
                        continue
                    token = id_to_token.get(token_id, "")
                    if not self._is_word_token(token):
                        continue
                    candidate = self._normalize(token, source_word)
                    if candidate.lower() == source_word.lower() or not self._morphology_ok(source_word, candidate):
                        continue
                    margin = float(score) - original_score
                    if margin < self.min_margin:
                        continue
                    if best is None or margin > best[0]:
                        best = (margin, candidate)
                if best is None:
                    continue
                margin, candidate = best
                out.append(EditCandidate(
                    source_word,
                    candidate,
                    min(0.91, 0.68 + margin * 0.06),
                    "russian-mlm",
                    f"ruBert masked-token margin={margin:.2f}",
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
