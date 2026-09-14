"""High-precision rules for government and predicate-boundary punctuation."""
from __future__ import annotations

import re

from decision_engine import EditCandidate
from morphology import Morphology, get_morphology, preserve_capitalization, preserve_yo

WORD = r"[А-Яа-яЁё]+"
ORDER_PROCESS_RE = re.compile(
    rf"\bпорядк[А-Яа-яЁё]*\s+(?P<process>{WORD}ний)\s+(?P<object>{WORD})\b",
    re.IGNORECASE,
)
DOCUMENT_TARGET_RE = re.compile(
    rf"\bв\s+(?P<target>раздел|подраздел|пункт|подпункт|графу|таблицу)"
    rf"(?P<number>\s+(?:№\s*)?\d+)\b",
    re.IGNORECASE,
)
LOCATIVE_PREDICATE_RE = re.compile(
    rf"\b(?:в|на)\s+{WORD}(?:\s+{WORD}){{0,3}}(?P<comma>,)(?P<gap>\s+)"
    rf"(?P<predicate>{WORD})\b",
    re.IGNORECASE,
)

PROCESS_LEMMAS = frozenset({
    "ведение", "проведение", "выполнение", "осуществление", "оформление",
    "предоставление", "заполнение", "составление", "хранение", "использование",
    "применение", "рассмотрение", "согласование", "утверждение", "планирование",
    "обеспечение", "отражение", "представление", "размещение", "формирование",
})
STATIVE_PREDICATES = re.compile(
    r"\b(?:не\s+)?(?:выставляется|указывается|отражается|содержится|имеется|"
    r"приводится|представляется|фиксируется|заполняется)\b",
    re.IGNORECASE,
)
IMPERSONAL_PARTICIPLES = frozenset({
    "принято", "проведено", "оказано", "организовано", "установлено",
    "выявлено", "осуществлено", "выполнено", "рассмотрено", "подготовлено",
    "проверено", "отмечено", "обеспечено", "представлено", "определено",
})
LOCATIVE_FORMS = {
    "раздел": "разделе",
    "подраздел": "подразделе",
    "пункт": "пункте",
    "подпункт": "подпункте",
    "графу": "графе",
    "таблицу": "таблице",
}


class V16RuleExtension:
    def __init__(self, morphology: Morphology | None = None) -> None:
        self.morph = morphology or get_morphology()

    def _process_government(self, text: str) -> list[EditCandidate]:
        out: list[EditCandidate] = []
        for match in ORDER_PROCESS_RE.finditer(text):
            source = match.group("process")
            if not (self.morph.lemmas(source) & PROCESS_LEMMAS):
                continue
            forms = self.morph.inflected_forms(source, {"gent", "sing"})
            forms = {preserve_capitalization(source, preserve_yo(source, f)) for f in forms}
            forms = {f for f in forms if f.casefold() != source.casefold()}
            if len(forms) != 1:
                forms = {source[:-2] + "ия"} if source.casefold().endswith("ний") else set()
            if len(forms) != 1:
                continue
            target = forms.pop()
            out.append(EditCandidate(
                source, target, 0.995, "rule-process-government",
                "после «порядок» название процесса употребляется в родительном единственного числа",
                start=match.start("process"),
            ))
        return out

    @staticmethod
    def _document_locative(text: str) -> list[EditCandidate]:
        out: list[EditCandidate] = []
        for match in DOCUMENT_TARGET_RE.finditer(text):
            tail = text[match.end():match.end() + 320]
            if not STATIVE_PREDICATES.search(tail):
                continue
            source = match.group("target")
            target = preserve_capitalization(source, LOCATIVE_FORMS[source.casefold()])
            out.append(EditCandidate(
                source, target, 0.995, "rule-document-locative",
                "местонахождение в разделе документа требует предложного падежа",
                start=match.start("target"),
            ))
        return out

    @staticmethod
    def _comma_before_impersonal_predicate(text: str) -> list[EditCandidate]:
        out: list[EditCandidate] = []
        for match in LOCATIVE_PREDICATE_RE.finditer(text):
            if match.group("predicate").casefold() not in IMPERSONAL_PARTICIPLES:
                continue
            out.append(EditCandidate(
                ",", "", 0.997, "rule-predicate-boundary",
                "обстоятельство места не отделяется запятой от сказуемого",
                start=match.start("comma"),
            ))
        return out

    def candidates(self, text: str) -> list[EditCandidate]:
        if not text:
            return []
        return (
            self._process_government(text)
            + self._document_locative(text)
            + self._comma_before_impersonal_predicate(text)
        )
