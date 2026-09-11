"""Layout-aware punctuation processing for wrapped LibreOffice text."""
from __future__ import annotations

import difflib
import re

from decision_engine import EditCandidate
from reasoning_cascade import ReasoningCascade
from safe_diff import diff_candidates
from segmentation import split_sentences, strip_enumeration

TOKEN_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+|[^\sА-Яа-яЁёA-Za-z0-9]")
SOFT_BREAK_RE = re.compile(r"(?<!\n)\n(?!\n)")
WORD = r"[А-Яа-яЁё]+"
ORG_RE = re.compile(
    rf"(?P<participle>{WORD})\s+с\s+личн(?:ым|ого|ому|ом)\s+состав(?:ом|а|у|е)"
    rf"(?P<badcomma>\s*,)(?P<space>\s+)"
    rf"(?P<org>(?:ООО|АО|ПАО|ФГБУ|ФГУП)\s*«[^»]+»)"
    rf"(?P<gap>\s+)(?P<next>{WORD})",
    re.IGNORECASE,
)


def collapse_soft_breaks(text: str) -> str:
    """Turn visual line wraps into spaces, preserving real blank paragraphs."""
    return SOFT_BREAK_RE.sub(" ", text)


def restore_soft_breaks(original: str, corrected: str) -> str:
    """Project original visual wraps onto corrected text by token alignment."""
    breaks = [m.start() for m in SOFT_BREAK_RE.finditer(original)]
    if not breaks:
        return corrected
    src = [(m.group(0), m.start()) for m in TOKEN_RE.finditer(original)]
    dst = [(m.group(0), m.start()) for m in TOKEN_RE.finditer(corrected)]
    if not src or not dst:
        return corrected
    matcher = difflib.SequenceMatcher(
        None, [t[0].casefold() for t in src], [t[0].casefold() for t in dst],
        autojunk=False,
    )
    index_map: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for shift in range(i2 - i1):
                index_map[i1 + shift] = j1 + shift
    insertions: list[int] = []
    for position in breaks:
        right = next((i for i, (_, start) in enumerate(src) if start > position), None)
        if right is None:
            continue
        mapped = index_map.get(right)
        if mapped is not None:
            insertions.append(dst[mapped][1])
    result = corrected
    for position in sorted(set(insertions), reverse=True):
        left = position
        while left > 0 and result[left - 1] in " \t":
            left -= 1
        result = result[:left] + "\n" + result[position:]
    return result


class OfficePunctuationRules:
    """High-precision rules for official documents with company names."""

    def __init__(self, morphology) -> None:
        self.morph = morphology

    def candidates(self, text: str) -> list[EditCandidate]:
        out: list[EditCandidate] = []
        for match in ORG_RE.finditer(text):
            participle = match.group("participle")
            following = match.group("next")
            if not any(p.tag.POS == "PRTF" for p in self.morph.attributive_parses(participle)):
                continue
            if not self.morph.attributive_parses(following):
                continue
            comma_pos = match.start("badcomma") + match.group("badcomma").rfind(",")
            out.append(EditCandidate(
                ",", "", 0.995, "rule-punctuation",
                "запятая разрывает дополнение с наименованием организации",
                start=comma_pos,
            ))
            org = match.group("org")
            out.append(EditCandidate(
                org, org + ",", 0.995, "rule-punctuation",
                "закрытие причастного оборота перед определяемым существительным",
                start=match.start("org"),
            ))
        return out


class LayoutAwareReasoningCascade(ReasoningCascade):
    """Run models on unwrapped text, then restore LibreOffice line layout."""

    async def candidates(self, text: str, context: str = "") -> list[EditCandidate]:
        if not self.enabled:
            return []
        out: list[EditCandidate] = []
        for segment in [strip_enumeration(s) for s in split_sentences(text)][:self.max_sentences]:
            model_text = collapse_soft_breaks(segment.text)
            try:
                corrected_flat = await self.correct(model_text, context)
                corrected = restore_soft_breaks(segment.text, corrected_flat)
            except Exception:
                self._failures += 1
                continue
            out.extend(diff_candidates(
                segment.text, corrected, "russian-gec-reasoning", 0.86,
                offset=segment.start,
            ))
        return out
