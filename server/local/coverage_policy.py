"""Coverage-aware escalation policy for local correction cascades."""
from __future__ import annotations

import os
import re

from decision_engine import EditCandidate

SKELETON_RE = re.compile(r"[А-Яа-яЁёA-Za-z]+|\d+")


def is_punctuation_only(candidate: EditCandidate) -> bool:
    """True when an edit changes only punctuation or spacing."""
    return SKELETON_RE.findall(candidate.before) == SKELETON_RE.findall(candidate.after)


def needs_deep_review(text: str, candidates: list[EditCandidate]) -> bool:
    """Escalate when the fast pass found no substantive language correction.

    The old `if fast: stop` policy treated one comma as proof that the whole
    selection was checked. This is a coverage decision, not an error-count
    decision: punctuation-only candidates must not suppress grammar review.
    """
    min_chars = int(os.getenv("REASONING_COVERAGE_MIN_CHARS", "50"))
    if len(text.strip()) < min_chars:
        return not candidates
    return not candidates or all(is_punctuation_only(c) for c in candidates)


def needs_rescue_despite_verified_punctuation(
    candidates: list[EditCandidate], original_decision: bool,
) -> bool:
    """Do not let a verified comma suppress A/B/Y rescue models."""
    if candidates and all(is_punctuation_only(c) for c in candidates):
        return True
    return original_decision
