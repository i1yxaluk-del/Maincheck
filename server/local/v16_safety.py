"""Precision guard for unsupported punctuation emitted by generative stages."""
from __future__ import annotations

from dataclasses import replace

from decision_engine import EditCandidate
from verification import is_generative

QUOTE_CHARS = frozenset('«»"“”„‟')
SOLO_PUNCTUATION_CONFIDENCE = 0.49


def _quote_signature(value: str) -> tuple[str, ...]:
    return tuple(ch for ch in value if ch in QUOTE_CHARS)


def suppress_unsafe_solo_punctuation(
    candidates: list[EditCandidate],
) -> list[EditCandidate]:
    """Lower unsupported model punctuation below the decision threshold.

    RuPunct-backed candidates are exempt. Deletions remain allowed because
    the production failures were unsupported additions and asymmetric quote
    substitutions, while local punctuation removal is a useful SAGE signal.
    """
    out: list[EditCandidate] = []
    for candidate in candidates:
        if not is_generative(candidate.category):
            out.append(candidate)
            continue
        sources = candidate.sources or (candidate.category,)
        has_rupunct = any("rupunct" in source for source in sources)
        risky_quote = _quote_signature(candidate.before) != _quote_signature(candidate.after)
        solo_comma_addition = (
            not has_rupunct
            and len(sources) == 1
            and candidate.after.count(",") > candidate.before.count(",")
        )
        if risky_quote or solo_comma_addition:
            out.append(replace(
                candidate,
                confidence=min(candidate.confidence, SOLO_PUNCTUATION_CONFIDENCE),
            ))
        else:
            out.append(candidate)
    return out
