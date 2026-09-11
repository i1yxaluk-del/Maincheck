"""Targeted high-precision regressions discovered after the v9 rollout.

LibreOffice sends visual line wraps as newlines.  Rules in this module therefore
use ``\s+`` and preserve the original layout while editing only the offending
word or comma.
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


class V10RuleExtension:
    """Conservative deterministic rules missing from the v9 corpus."""

    def __init__(self, morphology: Morphology | None = None) -> None:
        self.morph = morphology or get_morphology()

    @property
    def available(self) -> bool:
        return self.morph.available

    def _plural_head_after_genitive_prep(self, text: str) -> list[EditCandidate]:
        """``после ночных наряда`` -> ``после ночных нарядов``.

        We act only when the modifier is unambiguously plural in the genitive,
        is not also a noun (protects ``после данных анализа``), while the head
        is unambiguously singular in the genitive and has one dictionary plural
        genitive form.  Whitespace may contain visual line wraps.
        """
        out: list[EditCandidate] = []
        for match in PREP_NUMBER_RE.finditer(text):
            modifier = match.group("modifier")
            head = match.group("head")
            if self.morph.noun_parses(modifier):
                continue
            modifier_parses = [
                p for p in self.morph.attributive_parses(modifier)
                if p.tag.case == "gent"
            ]
            if not modifier_parses or {p.tag.number for p in modifier_parses} != {"plur"}:
                continue
            head_parses = [p for p in self.morph.noun_parses(head) if p.tag.case == "gent"]
            if not head_parses or {p.tag.number for p in head_parses} != {"sing"}:
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
                head,
                fixed,
                0.992,
                "rule-prep-number",
                f"согласование числа после предлога «{match.group('prep')}»",
                start=match.start("head"),
            ))
        return out

    def _comma_inside_fact_complement(self, text: str) -> list[EditCandidate]:
        """Remove a comma splitting ``факты заступления на ...``.

        The dependent must be a deverbal noun in genitive singular and must be
        followed by its own prepositional complement.  These restrictions avoid
        treating an ordinary enumeration after ``факты`` as a genitive chain.
        """
        out: list[EditCandidate] = []
        for match in FACT_COMPLEMENT_RE.finditer(text):
            word = match.group("dependent")
            parses = [
                p for p in self.morph.noun_parses(word)
                if p.tag.case == "gent" and p.tag.number == "sing"
                and p.normal_form.endswith(("ние", "тие"))
            ]
            if not parses:
                continue
            comma = match.start("comma") + match.group("comma").rfind(",")
            out.append(EditCandidate(
                ",",
                "",
                0.99,
                "rule-punctuation",
                "запятая разрывает сочетание «факты + родительный падеж»",
                start=comma,
            ))
        return out

    def candidates(self, text: str) -> list[EditCandidate]:
        if not self.available or not text:
            return []
        return self._plural_head_after_genitive_prep(text) + self._comma_inside_fact_complement(text)
