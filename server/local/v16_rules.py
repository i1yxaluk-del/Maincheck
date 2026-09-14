"""Высокоточные правила управления, согласования и пунктуации."""
from __future__ import annotations

import re

from decision_engine import EditCandidate
from morphology import (
    Morphology,
    features,
    get_morphology,
    preserve_capitalization,
    preserve_yo,
)

WORD = r"[А-Яа-яЁё]+"
WORD_RE = re.compile(WORD)
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
    "раздел": "разделе", "подраздел": "подразделе", "пункт": "пункте",
    "подпункт": "подпункте", "графу": "графе", "таблицу": "таблице",
}
ADJECTIVE_ENDINGS = (
    "ый", "ий", "ой", "ая", "яя", "ое", "ее", "ые", "ие", "ого", "его",
    "ому", "ему", "ую", "юю", "ым", "им", "ых", "их", "ыми", "ими",
)
PLURAL_CASE_ENDINGS = {
    "gent": ("ых", "их"),
    "loct": ("ых", "их"),
    "datv": ("ым", "им"),
    "ablt": ("ыми", "ими"),
    "nomn": ("ые", "ие"),
}


class V16RuleExtension:
    """Общие детерминированные правила для официально-делового текста."""

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
            out.append(EditCandidate(
                source, forms.pop(), 0.995, "rule-process-government",
                "после «порядок» название процесса употребляется в родительном единственного числа",
                start=match.start("process"),
            ))
        return out

    @staticmethod
    def _looks_like_modifier(word: str) -> bool:
        return word.casefold().endswith(ADJECTIVE_ENDINGS)

    @staticmethod
    def _surface_agrees(word: str, head_features) -> bool:
        """Даёт голос неизвестному словарю при однозначном окончании."""
        if head_features.number != "plur":
            return False
        endings = PLURAL_CASE_ENDINGS.get(head_features.case)
        return bool(endings and word.casefold().endswith(endings))

    def _modifier_chain_agreement(self, text: str) -> list[EditCandidate]:
        """Находит один выбивающийся модификатор в цепочке перед существительным.

        Исправление создаётся только при поддержке минимум двух соседних
        определений и единственном результате словоизменения. Неизвестное
        словарю определение может подтвердить форму своим русским окончанием,
        но само по такому эвристическому разбору никогда не исправляется.
        """
        words = list(WORD_RE.finditer(text))
        out: list[EditCandidate] = []
        used: set[int] = set()
        for head_index in range(2, len(words)):
            head_match = words[head_index]
            head_parses = self.morph.noun_parses(head_match.group(0))
            if not head_parses:
                continue
            modifiers = []
            cursor = head_index - 1
            while cursor >= 0 and len(modifiers) < 4:
                current = words[cursor]
                gap = text[current.end():words[cursor + 1].start()]
                parses = self.morph.attributive_parses(current.group(0))
                if not gap or not gap.isspace():
                    break
                if not parses and not self._looks_like_modifier(current.group(0)):
                    break
                modifiers.insert(0, (current, parses))
                cursor -= 1
            if len(modifiers) < 3:
                continue

            proposals: set[tuple[int, str]] = set()
            for head_parse in head_parses[:8]:
                head_features = features(head_parse)
                mismatches = []
                for position, (modifier_match, parses) in enumerate(modifiers):
                    agrees = (
                        any(features(p).agrees_with(head_features) for p in parses)
                        if parses else self._surface_agrees(modifier_match.group(0), head_features)
                    )
                    if not agrees:
                        mismatches.append((position, modifier_match, parses))
                if len(mismatches) != 1 or len(modifiers) - 1 < 2:
                    continue
                _, mismatch, parses = mismatches[0]
                if not parses:
                    continue
                source = mismatch.group(0)
                target = self.morph.inflect_modifier(source, head_parse)
                if target and target.casefold() != source.casefold():
                    proposals.add((mismatch.start(), target))
            if len(proposals) != 1:
                continue
            start, target = proposals.pop()
            if start in used:
                continue
            source = next(w.group(0) for w in words if w.start() == start)
            used.add(start)
            out.append(EditCandidate(
                source, target, 0.992, "rule-modifier-chain-agreement",
                f"форма определения должна согласовываться с существительным «{head_match.group(0)}»",
                start=start,
            ))
        return out

    @staticmethod
    def _document_locative(text: str) -> list[EditCandidate]:
        out: list[EditCandidate] = []
        for match in DOCUMENT_TARGET_RE.finditer(text):
            if not STATIVE_PREDICATES.search(text[match.end():match.end() + 320]):
                continue
            source = match.group("target")
            out.append(EditCandidate(
                source, preserve_capitalization(source, LOCATIVE_FORMS[source.casefold()]),
                0.995, "rule-document-locative",
                "местонахождение в разделе документа требует предложного падежа",
                start=match.start("target"),
            ))
        return out

    @staticmethod
    def _comma_before_impersonal_predicate(text: str) -> list[EditCandidate]:
        out: list[EditCandidate] = []
        for match in LOCATIVE_PREDICATE_RE.finditer(text):
            if match.group("predicate").casefold() in IMPERSONAL_PARTICIPLES:
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
            + self._modifier_chain_agreement(text)
            + self._document_locative(text)
            + self._comma_before_impersonal_predicate(text)
        )
