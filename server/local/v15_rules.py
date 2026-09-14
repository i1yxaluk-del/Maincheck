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
    """Lexically general rules; every edit is morphology- or ending-proven."""

    def __init__(self, morphology: Morphology | None = None) -> None:
        self.morph = morphology or get_morphology()

    @staticmethod
    def _instrumental_plural_surface(suffix: str, head: str) -> str | None:
        """Correct -ым/-им before a noun ending in plural instrumental -ами/-ями."""
        if not head.casefold().endswith(("ами", "ями")):
            return None
        lower = suffix.casefold()
        if lower.endswith("ым"):
            return suffix[:-2] + ("ЫМИ" if suffix[-2:].isupper() else "ыми")
        if lower.endswith("им"):
            return suffix[:-2] + ("ИМИ" if suffix[-2:].isupper() else "ими")
        return None

    def _compound_modifier_agreement(self, text: str) -> list[EditCandidate]:
        out: list[EditCandidate] = []
        for match in COMPOUND_MODIFIER_RE.finditer(text):
            modifier = match.group("modifier")
            head = match.group("head")
            prefix, suffix = modifier.rsplit("-", 1)

            surface = self._instrumental_plural_surface(suffix, head)
            if surface:
                produced_suffixes = {surface}
            else:
                head_parses = self.morph.noun_parses(head)
                suffix_parses = self.morph.attributive_parses(suffix)
                if not head_parses or not suffix_parses or self.morph.pair_agrees(suffix, head):
                    continue
                produced_suffixes: set[str] = set()
                for head_parse in head_parses[:8]:
                    fixed_suffix = self.morph.inflect_modifier(suffix, head_parse)
                    if fixed_suffix:
                        produced_suffixes.add(fixed_suffix)

            produced_suffixes = {
                p for p in produced_suffixes if p.casefold() != suffix.casefold()
            }
            if len(produced_suffixes) != 1:
                continue
            fixed_suffix = preserve_capitalization(
                suffix, preserve_yo(suffix, produced_suffixes.pop())
            )
            # Change only the final component. Apart from preserving Writer
            # formatting, this intentionally keeps the protected compound-term
            # guard active against lexical replacement of the whole compound.
            suffix_start = match.start("modifier") + len(prefix) + 1
            out.append(EditCandidate(
                suffix, fixed_suffix, 0.988, "rule-compound-agreement",
                f"согласование составного определения с существительным «{head}»",
                start=suffix_start,
            ))
        return out

    def _coordinated_modifier_comma(self, text: str) -> list[EditCandidate]:
        out: list[EditCandidate] = []
        for match in COORDINATED_MODIFIERS_RE.finditer(text):
            first, second, head = match.group("first"), match.group("second"), match.group("head")
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
            comma_pos = match.start("comma") + match.group("comma").rfind(",")
            before = text[match.start("first"):comma_pos + 1]
            out.append(EditCandidate(
                before, before[:-1], 0.995, "rule-coordinated-modifiers",
                "перед одиночным союзом между однородными определениями запятая не ставится",
                start=match.start("first"),
            ))
        return out

    def candidates(self, text: str) -> list[EditCandidate]:
        if not text:
            return []
        return self._compound_modifier_agreement(text) + self._coordinated_modifier_comma(text)
