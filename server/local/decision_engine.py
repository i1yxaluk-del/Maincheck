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
    """Conservative merger between LLM edit candidates and deterministic guards."""

    def __init__(self, min_confidence: float = 0.55, max_changes: int = 40,
                 max_before_chars: int = 180, protected_words: set[str] | None = None) -> None:
        self.min_confidence = min_confidence
        self.max_changes = max_changes
        self.max_before_chars = max_before_chars
        self.protected_words = {w.casefold() for w in (protected_words or set()) if w}

    @staticmethod
    def parse(payload: str | dict[str, Any]) -> list[EditCandidate]:
        data = payload if isinstance(payload, dict) else json.loads(payload.strip().strip("`"))
        raw = data.get("edits", []) if isinstance(data, dict) else []
        if not isinstance(raw, list):
            return []
        out: list[EditCandidate] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            before, after = item.get("before"), item.get("after")
            if not isinstance(before, str) or not isinstance(after, str):
                continue
            try:
                confidence = float(item.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 0.0
            out.append(EditCandidate(
                before, after, max(0.0, min(1.0, confidence)),
                str(item.get("category", "unknown")), str(item.get("reason", "")),
            ))
        return out

    def _protected(self, before: str) -> bool:
        if not self.protected_words:
            return False
        tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9_-]*", before)
        return any(t.casefold() in self.protected_words for t in tokens)

    def validate(self, text: str, candidates: list[EditCandidate]) -> list[tuple[int, EditCandidate]]:
        accepted: list[tuple[int, EditCandidate]] = []
        occupied: list[tuple[int, int]] = []
        for c in sorted(candidates, key=lambda x: (-x.confidence, -len(x.before))):
            if len(accepted) >= self.max_changes or not c.before or c.before == c.after:
                continue
            if len(c.before) > self.max_before_chars or c.confidence < self.min_confidence:
                continue
            if c.before.replace("ё", "е").replace("Ё", "Е") == c.after.replace("ё", "е").replace("Ё", "Е"):
                continue
            if self._protected(c.before):
                continue
            # Ambiguous BEFORE text cannot be safely mapped to one occurrence.
            positions = [m.start() for m in re.finditer(re.escape(c.before), text)]
            if len(positions) != 1:
                continue
            start, end = positions[0], positions[0] + len(c.before)
            if any(not (end <= a or start >= b) for a, b in occupied):
                continue
            occupied.append((start, end))
            accepted.append((start, c))
        return sorted(accepted, key=lambda x: x[0], reverse=True)

    def apply(self, text: str, candidates: list[EditCandidate]) -> tuple[str, list[EditCandidate]]:
        accepted_spans = self.validate(text, candidates)
        result = text
        for start, c in accepted_spans:
            result = result[:start] + c.after + result[start + len(c.before):]
        return result, [c for _, c in reversed(accepted_spans)]

    @staticmethod
    def diff_candidates(original: str, corrected: str) -> list[EditCandidate]:
        if original == corrected:
            return []
        sm = difflib.SequenceMatcher(a=original, b=corrected, autojunk=False)
        return [EditCandidate(original[i1:i2], corrected[j1:j2], 1.0, "diff", "server diff")
                for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]
