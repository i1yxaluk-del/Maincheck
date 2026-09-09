from __future__ import annotations

import difflib

from decision_engine import EditCandidate


def diff_candidates(source: str, corrected: str, category: str, confidence: float = 0.70) -> list[EditCandidate]:
    if not source or not corrected or source == corrected:
        return []
    if source.count("\n") != corrected.count("\n"):
        return []
    if len(source.split("\n\n")) != len(corrected.split("\n\n")):
        return []
    out: list[EditCandidate] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, source, corrected, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if "\n" in source[i1:i2] or "\n" in corrected[j1:j2]:
            return []
        if tag == "replace" and source[i1:i2] and corrected[j1:j2]:
            before, after = source[i1:i2], corrected[j1:j2]
        elif tag in {"insert", "delete"}:
            # Convert insertion/deletion into a small anchored replacement so
            # DecisionEngine can map it to one exact occurrence. This is how
            # punctuation such as `требует корректировки` ->
            # `требует, корректировки` survives without permitting empty-BEFORE edits.
            left_s, left_c = max(0, i1 - 12), max(0, j1 - 12)
            right_s, right_c = min(len(source), i2 + 12), min(len(corrected), j2 + 12)
            before, after = source[left_s:right_s], corrected[left_c:right_c]
        else:
            return []
        if not before or not after or len(before) > 70 or len(after) > 70:
            return []
        if before == after:
            continue
        out.append(EditCandidate(before, after, confidence, category, "bounded contextual diff"))
    return out
