"""Высокоточные правила для новых классов ошибок официального текста."""
from __future__ import annotations

import re

from decision_engine import EditCandidate
from morphology import Morphology, get_morphology, preserve_capitalization, preserve_yo

WORD = r"[А-Яа-яЁё]+"
PRI_PROCESS_RE = re.compile(
    rf"\bпри\s+(?P<process>{WORD}ниях)\s+(?P<object>{WORD})\b",
    re.IGNORECASE,
)
COURSE_PREDICATE_RE = re.compile(
    rf"\bв\s+ходе\s+{WORD}(?P<comma>,)(?P<gap>\s+)(?P<verb>{WORD})\b",
    re.IGNORECASE,
)
PROCESS_LEMMAS = frozenset({
    "несение", "ведение", "проведение", "выполнение", "осуществление",
    "оформление", "заполнение", "составление", "рассмотрение",
    "согласование", "утверждение", "планирование", "обеспечение",
    "представление", "размещение", "формирование",
})


class V17RuleExtension:
    """Правила, которые исправляют класс конструкций, а не отдельные фразы."""

    def __init__(self, morphology: Morphology | None = None) -> None:
        self.morph = morphology or get_morphology()

    def _process_after_pri(self, text: str) -> list[EditCandidate]:
        out: list[EditCandidate] = []
        for match in PRI_PROCESS_RE.finditer(text):
            source = match.group("process")
            if not (self.morph.lemmas(source) & PROCESS_LEMMAS):
                continue
            forms = self.morph.inflected_forms(source, {"loct", "sing"})
            forms = {
                preserve_capitalization(source, preserve_yo(source, form))
                for form in forms
                if form.casefold() != source.casefold()
            }
            if len(forms) != 1:
                continue
            out.append(EditCandidate(
                source, forms.pop(), 0.995, "rule-process-after-pri",
                "после «при» название одного процесса употребляется в предложном единственного числа",
                start=match.start("process"),
            ))
        return out

    def _comma_after_course_phrase(self, text: str) -> list[EditCandidate]:
        out: list[EditCandidate] = []
        for match in COURSE_PREDICATE_RE.finditer(text):
            parses = self.morph.parses_with_pos(match.group("verb"), {"VERB", "INFN"})
            if not parses:
                continue
            out.append(EditCandidate(
                ",", "", 0.995, "rule-course-predicate-boundary",
                "обстоятельство «в ходе …» не отделяется запятой от сказуемого",
                start=match.start("comma"),
            ))
        return out

    def candidates(self, text: str) -> list[EditCandidate]:
        if not text:
            return []
        return self._process_after_pri(text) + self._comma_after_course_phrase(text)
