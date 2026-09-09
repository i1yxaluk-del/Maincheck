from __future__ import annotations

import difflib
import math
import re

from decision_engine import EditCandidate

WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z]+")


def _global_rewrite(source: str, corrected: str) -> bool:
    """Reject paraphrases/reordering; allow only a small number of local edits."""
    source_words = WORD_RE.findall(source)
    corrected_words = WORD_RE.findall(corrected)
    if source_words and abs(len(corrected_words) - len(source_words)) > 1:
        return True
    baseline = max(1, min(len(source), len(corrected)))
    if abs(len(corrected) - len(source)) / baseline > 0.20:
        return True

    matcher = difflib.SequenceMatcher(None, source_words, corrected_words, autojunk=False)
    changed = 0
    replace_ops = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed += max(i2 - i1, j2 - j1)
        if tag == "replace":
            replace_ops += 1
        if (i2 - i1) > 2 or (j2 - j1) > 2:
            return True
    allowed = max(3, math.ceil(max(len(source_words), 1) * 0.12))
    return changed > allowed or replace_ops > 3


def diff_candidates(source: str, corrected: str, category: str, confidence: float = 0.70) -> list[EditCandidate]:
    if not source or not corrected or source == corrected:
        return []
    if source.count("\n") != corrected.count("\n"):
        return []
    if len(source.split("\n\n")) != len(corrected.split("\n\n")):
        return []
    if _global_rewrite(source, corrected):
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
