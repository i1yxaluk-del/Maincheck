"""High-precision rules for regressions found after the v9 rollout.

LibreOffice may send visual line wraps as newlines.  Patterns therefore use
``\s+`` and edits replace only the erroneous token or comma, preserving layout.
"""
from __future__ import annotations

import re

from decision_engine import EditCandidate
from morphology import Morphology, get_morphology, preserve_capitalization, preserve_yo

WORD = r"[А-Яа-яЁё]+"
GENITIVE_PREPS = (
    "без|для|до|из|из-за|из-под|от|у|кроме|вместо|вокруг|возле|около|"
    "после|против|среди|путем|путём|посредством|относительно|касательно|"
    "ввиду|вследствие|сверх|мимо|внутри|вне"
)
PREP_NUMBER_RE = re.compile(
    rf"\b(?P<prep>{GENITIVE_PREPS})\s+(?P<modifier>{WORD})\s+(?P<head>{WORD})\b",
    re.IGNORECASE,
)
FACT_COMPLEMENT_RE = re.compile(
    rf"\bфакты(?P<comma>\s*,)(?P<space>\s+)(?P<dependent>{WORD})"
    rf"(?=\s+(?:на|при|в|по|из|после|для|от)\b)",
    re.IGNORECASE,
)
SUBSTANTIVE_MODIFIERS = frozenset({"данных", "сведений"})


class V10RuleExtension:
    """Conservative deterministic rules missing from the v9 corpus."""

    def __init__(self, morphology: Morphology | None = None) -> None:
        self.morph = morphology or get_morphology()

    @property
    def available(self) -> bool:
        return self.morph.available

    def _plural_head_after_genitive_prep(self, text: str) -> list[EditCandidate]:
        """Correct ``после ночных наряда`` to ``после ночных нарядов``.

        The modifier must have a plural-genitive adjectival reading, without a
        singular-genitive adjectival reading.  The head must have a
        singular-genitive noun reading and one plural-genitive dictionary form.
        Known substantivized forms such as ``данных`` are excluded.
        """
        out: list[EditCandidate] = []
        for match in PREP_NUMBER_RE.finditer(text):
            modifier = match.group("modifier")
            head = match.group("head")
            if modifier.casefold() in SUBSTANTIVE_MODIFIERS:
                continue
            modifier_parses = [
                p for p in self.morph.attributive_parses(modifier)
                if p.tag.case == "gent"
            ]
            if not modifier_parses:
                continue
            numbers = {p.tag.number for p in modifier_parses}
            if "plur" not in numbers or "sing" in numbers:
                continue
            head_parses = [p for p in self.morph.noun_parses(head) if p.tag.case == "gent"]
            if not head_parses or not any(p.tag.number == "sing" for p in head_parses):
                continue
            if any(p.tag.number == "plur" for p in head_parses):
                continue
            produced: set[str] = set()
            for parse in head_parses[:8]:
                form = parse.inflect({"gent", "plur"})
                if form and form.word:
                    produced.add(form.word)
            produced = {word for word in produced if word.casefold() != head.casefold()}
            if len(produced) != 1:
                continue
            fixed = preserve_capitalization(head, preserve_yo(head, produced.pop()))
            out.append(EditCandidate(
                head, fixed, 0.992, "rule-prep-number",
                f"согласование числа после предлога «{match.group('prep')}»",
                start=match.start("head"),
            ))
        return out

    def _comma_inside_fact_complement(self, text: str) -> list[EditCandidate]:
        """Remove a comma splitting ``факты заступления на ...``.

        The dependent is restricted to a deverbal ``-ние/-тие`` noun followed
        by a prepositional complement.  This does not match an ordinary list
        such as ``факты, нарушения и недостатки``.
        """
        out: list[EditCandidate] = []
        for match in FACT_COMPLEMENT_RE.finditer(text):
            word = match.group("dependent")
            parses = [
                p for p in self.morph.noun_parses(word)
                if p.tag.case == "gent" and p.tag.number == "sing"
                and p.normal_form.endswith(("ние", "тие"))
            ]
            # Morphological dictionaries differ slightly between deployments;
            # the surface ending is a safe fallback under the strict context.
            surface_deverbal = word.casefold().endswith(("ния", "тия"))
            if not parses and not surface_deverbal:
                continue
            comma = match.start("comma") + match.group("comma").rfind(",")
            out.append(EditCandidate(
                ",", "", 0.99, "rule-punctuation",
                "запятая разрывает сочетание «факты + родительный падеж»",
                start=comma,
            ))
        return out

    def candidates(self, text: str) -> list[EditCandidate]:
        if not self.available or not text:
            return []
        return self._plural_head_after_genitive_prep(text) + self._comma_inside_fact_complement(text)
