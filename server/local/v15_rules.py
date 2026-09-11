"""General high-precision fixes for coordinated and compound modifiers."""
from __future__ import annotations

import re

from decision_engine import EditCandidate
from morphology import Morphology, features, get_morphology, preserve_capitalization, preserve_yo

WORD = r"[А-Яа-яЁё]+"
COMPOUND_MODIFIER_RE = re.compile(
    rf"\b(?P<modifier>{WORD}(?:-{WORD})+)(?P<gap>\s+)(?P<head>{WORD})\b"
)
COORDINATED_MODIFIERS_RE = re.compile(
    rf"\b(?P<first>{WORD})(?P<comma>\s*,)(?P<gap>\s+)"
    rf"(?P<conj>и|или|либо)(?P<gap2>\s+)(?P<second>{WORD})"
    rf"(?P<gap3>\s+)(?P<head>{WORD})\b",
    re.IGNORECASE,
)


class V15RuleExtension:
    """Lexically general rules; every edit is morphology-proven."""

    def __init__(self, morphology: Morphology | None = None) -> None:
        self.morph = morphology or get_morphology()

    def _compound_modifier_agreement(self, text: str) -> list[EditCandidate]:
        out: list[EditCandidate] = []
        for match in COMPOUND_MODIFIER_RE.finditer(text):
            modifier = match.group("modifier")
            head = match.group("head")
            suffix = modifier.rsplit("-", 1)[1]
            suffix_parses = self.morph.attributive_parses(suffix)
            head_parses = self.morph.noun_parses(head)
            if not suffix_parses or not head_parses:
                continue
            # Any valid reading protects the source from overcorrection.
            if self.morph.pair_agrees(suffix, head):
                continue
            produced: set[str] = set()
            for head_parse in head_parses[:8]:
                fixed_suffix = self.morph.inflect_modifier(suffix, head_parse)
                if fixed_suffix:
                    produced.add(modifier.rsplit("-", 1)[0] + "-" + fixed_suffix)
            produced = {p for p in produced if p.casefold() != modifier.casefold()}
            if len(produced) != 1:
                continue
            fixed = preserve_capitalization(modifier, preserve_yo(modifier, produced.pop()))
            out.append(EditCandidate(
                modifier, fixed, 0.988, "rule-compound-agreement",
                f"согласование составного определения с существительным «{head}»",
                start=match.start("modifier"),
            ))
        return out

    def _coordinated_modifier_comma(self, text: str) -> list[EditCandidate]:
        out: list[EditCandidate] = []
        for match in COORDINATED_MODIFIERS_RE.finditer(text):
            first, second, head = (
                match.group("first"), match.group("second"), match.group("head")
            )
            first_parses = self.morph.attributive_parses(first)
            second_parses = self.morph.attributive_parses(second)
            head_parses = self.morph.noun_parses(head)
            if not first_parses or not second_parses or not head_parses:
                continue
            same_group = any(
                features(a).agrees_with(features(noun))
                and features(b).agrees_with(features(noun))
                for a in first_parses for b in second_parses for noun in head_parses
            )
            if not same_group:
                continue
            # Anchor the deletion to the first modifier so the LibreOffice
            # client applies a local edit instead of replacing the paragraph.
            comma_pos = match.start("comma") + match.group("comma").rfind(",")
            before = text[match.start("first"):comma_pos + 1]
            after = before[:-1]
            out.append(EditCandidate(
                before, after, 0.995, "rule-coordinated-modifiers",
                "перед одиночным союзом между однородными определениями запятая не ставится",
                start=match.start("first"),
            ))
        return out

    def candidates(self, text: str) -> list[EditCandidate]:
        if not self.morph.available or not text:
            return []
        return self._compound_modifier_agreement(text) + self._coordinated_modifier_comma(text)
