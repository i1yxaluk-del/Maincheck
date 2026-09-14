"""Make accepted server edits safe for the legacy LibreOffice client.

The Basic client treats a change with an empty replacement (for example
`«,» -> «»`) as an unstructured response and falls back to replacing the whole
selection.  Expand punctuation deletions to a neighbouring lexical anchor so
both fragments are non-empty and Writer can apply a local range edit.
"""
from __future__ import annotations

from dataclasses import replace
import re

from decision_engine import EditCandidate

PUNCT_ONLY_RE = re.compile(r"^[\s,.;:!?—–-]+$")
LEFT_ANCHOR_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+(?:-[А-Яа-яЁёA-Za-z0-9]+)*\s*$")
RIGHT_ANCHOR_RE = re.compile(r"^\s*[А-Яа-яЁёA-Za-z0-9]+(?:-[А-Яа-яЁёA-Za-z0-9]+)*")


def materialize_client_safe_deletions(
    text: str, candidates: list[EditCandidate],
) -> list[EditCandidate]:
    """Turn punctuation -> empty into anchored local replacements."""
    out: list[EditCandidate] = []
    for candidate in candidates:
        if candidate.after or not PUNCT_ONLY_RE.fullmatch(candidate.before):
            out.append(candidate)
            continue
        start = candidate.start
        if start is None or text[start:start + len(candidate.before)] != candidate.before:
            out.append(candidate)
            continue
        left = LEFT_ANCHOR_RE.search(text[:start])
        if left is not None:
            anchor = left.group(0)
            out.append(replace(
                candidate,
                before=anchor + candidate.before,
                after=anchor,
                start=left.start(),
            ))
            continue
        right = RIGHT_ANCHOR_RE.match(text[start + len(candidate.before):])
        if right is not None:
            anchor = right.group(0)
            out.append(replace(
                candidate,
                before=candidate.before + anchor,
                after=anchor,
                start=start,
            ))
            continue
        out.append(candidate)
    return out
