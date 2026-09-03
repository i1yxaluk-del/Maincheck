from __future__ import annotations

from dataclasses import dataclass
import difflib
import json
import re
from typing import Any


@dataclass(frozen=True)
class EditCandidate:
    before: str
    after: str
    confidence: float = 1.0
    category: str = "unknown"
    reason: str = ""


class DecisionEngine:
    """Conservative merger between LLM edit candidates and deterministic guards.

    The engine never asks a model to produce the final document. It validates
    candidate edits against the original text, rejects overlaps and protected
    terms, then applies accepted edits from right to left. The legacy server
    can still render its ===CORRECTED===/===CHANGES=== protocol afterwards.
    """

    def __init__(
        self,
        min_confidence: float = 0.55,
        max_changes: int = 40,
        max_before_chars: int = 180,
        protected_words: set[str] | None = None,
    ) -> None:
        self.min_confidence = min_confidence
        self.max_changes = max_changes
        self.max_before_chars = max_before_chars
        self.protected_words = {w.casefold() for w in (protected_words or set()) if w}

    @staticmethod
    def parse(payload: str | dict[str, Any]) -> list[EditCandidate]:
        if isinstance(payload, dict):
            data = payload
        else:
            text = payload.strip()
            # Be tolerant of accidental markdown fences despite JSON schema.
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
            data = json.loads(text)
        raw = data.get("edits", []) if isinstance(data, dict) else []
        if not isinstance(raw, list):
            return []
        out: list[EditCandidate] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            before = item.get("before")
            after = item.get("after")
            if not isinstance(before, str) or not isinstance(after, str):
                continue
            try:
                confidence = float(item.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 0.0
            out.append(EditCandidate(
                before=before,
                after=after,
                confidence=max(0.0, min(1.0, confidence)),
                category=str(item.get("category", "unknown")),
                reason=str(item.get("reason", "")),
            ))
        return out

    @staticmethod
    def _span(text: str, needle: str, start: int = 0) -> tuple[int, int] | None:
        pos = text.find(needle, start)
        return None if pos < 0 else (pos, pos + len(needle))

    def _protected(self, before: str) -> bool:
        if not self.protected_words:
            return False
        tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9_-]*", before)
        return any(t.casefold() in self.protected_words for t in tokens)

    def validate(self, text: str, candidates: list[EditCandidate]) -> list[EditCandidate]:
        accepted: list[EditCandidate] = []
        occupied: list[tuple[int, int]] = []
        cursor = 0
        # Deterministic ordering makes identical requests reproducible.
        ranked = sorted(candidates, key=lambda x: (-x.confidence, -len(x.before)))
        for c in ranked:
            if len(accepted) >= self.max_changes:
                break
            if not c.before or c.before == c.after:
                continue
            if len(c.before) > self.max_before_chars or c.confidence < self.min_confidence:
                continue
            # Pure ё/е substitutions are stylistic normalization, not GEC.
            if c.before.replace("ё", "е").replace("Ё", "Е") == c.after.replace("ё", "е").replace("Ё", "Е"):
                continue
            if self._protected(c.before):
                continue
            span = self._span(text, c.before, cursor)
            if span is None:
                span = self._span(text, c.before)
            if span is None:
                continue
            if any(not (span[1] <= a or span[0] >= b) for a, b in occupied):
                continue
            occupied.append(span)
            accepted.append(c)
            cursor = span[1]
        return sorted(accepted, key=lambda x: text.find(x.before), reverse=True)

    def apply(self, text: str, candidates: list[EditCandidate]) -> tuple[str, list[EditCandidate]]:
        accepted = self.validate(text, candidates)
        result = text
        for c in accepted:
            pos = result.find(c.before)
            if pos >= 0:
                result = result[:pos] + c.after + result[pos + len(c.before):]
        return result, list(reversed(accepted))

    @staticmethod
    def diff_candidates(original: str, corrected: str) -> list[EditCandidate]:
        if original == corrected:
            return []
        sm = difflib.SequenceMatcher(a=original, b=corrected, autojunk=False)
        out: list[EditCandidate] = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            before, after = original[i1:i2], corrected[j1:j2]
            if before or after:
                out.append(EditCandidate(before, after, 1.0, "diff", "server diff"))
        return out
