from __future__ import annotations

import logging
import os
import re

from decision_engine import EditCandidate
from morphology import (
    Morphology,
    features,
    get_morphology,
    preserve_capitalization,
    preserve_yo,
)
from np_agreement import (
    CLAUSE_BREAK as CLAUSE_BREAK_CHARS,
    COORDINATORS,
    GENITIVE_QUANTIFIERS,
    NEGATIONS,
    NounPhraseAgreement,
)

log = logging.getLogger("ai_suggester.local_rules")

WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z]+(?:[-/][А-Яа-яЁёA-Za-z]+)*")
YEAR_RE = re.compile(r"\b(\d{4})\s+([А-Яа-яЁё-]+)\s+(год(?:а|у|ом|е|ов|ы)?|лет)\b")
PO_ONE_RE = re.compile(r"\bпо\s+одна\b", re.IGNORECASE)
REQUIRES_COMMA_RE = re.compile(r"\bтребует,\s+(?P<word>[А-Яа-яЁё]+)")
SEVERAL_BAD_RE = re.compile(r"\bнесколького\s+(?P<noun>[А-Яа-яЁё-]+)")
SEVERAL_GEN_RE = re.compile(r"\bнескольких\s+(?P<noun>[А-Яа-яЁё-]+)")
ONE_OR_SEVERAL_RE = re.compile(r"\bодного\s+или\s+нескольких\s+(?P<noun>[А-Яа-яЁё-]+)")


class LocalRuleEngine:
    """Детерминированные кандидаты там, где морфология даёт доказательство.

    Главное отличие v9 от v8 — правило согласования больше не «ищет
    отличающуюся форму», а требует морфологического доказательства
    ошибки: правка предлагается только если **ни одно** прочтение пары
    «определение + вершина» не является согласованным. Подробности и
    список закрытых классов ложных срабатываний — в `np_agreement.py`.
    """

    def __init__(self, morphology: Morphology | None = None) -> None:
        self.morph_helper = morphology or get_morphology()
        self.morph = self.morph_helper._morph if self.morph_helper.available else None
        self.agreement_enabled = os.getenv("RULE_AGREEMENT_ENABLED", "true").lower() in {
            "1", "true", "yes", "on",
        }
        self.predicative_enabled = os.getenv("RULE_PREDICATIVE_ENABLED", "true").lower() in {
            "1", "true", "yes", "on",
        }
        self.np_agreement = NounPhraseAgreement(self.morph_helper)

    @property
    def available(self) -> bool:
        return self.morph_helper.available

    # ------------------------------------------------------------------
    # Служебное
    # ------------------------------------------------------------------
    def _genitive_plural(self, word: str) -> str | None:
        """Родительный падеж множественного числа для существительного."""
        forms = self.morph_helper.inflected_forms(word, {"gent", "plur"})
        for form in forms:
            return preserve_yo(word, form)
        return None

    # ------------------------------------------------------------------
    # Правила управления количественных слов
    # ------------------------------------------------------------------
    def _quantifier_government(self, text: str) -> list[EditCandidate]:
        if not self.available:
            return []
        out: list[EditCandidate] = []

        for match in SEVERAL_BAD_RE.finditer(text):
            noun = match.group("noun")
            form = self._genitive_plural(noun)
            if not form:
                continue
            before = match.group(0)
            after = f"нескольких {form}"
            if before != after:
                out.append(EditCandidate(
                    before, after, 0.998, "rule-quantifier",
                    "«несколько» с существительным в родительном множественного числе",
                    start=match.start(),
                ))

        for match in ONE_OR_SEVERAL_RE.finditer(text):
            noun = match.group("noun")
            form = self._genitive_plural(noun) or noun
            before = match.group(0)
            after = f"одного или нескольких {form}"
            if before != after:
                out.append(EditCandidate(
                    before, after, 0.998, "rule-quantifier",
                    "конструкция «одного или нескольких» требует родительного множественного числа",
                    start=match.start(),
                ))

        for match in SEVERAL_GEN_RE.finditer(text):
            noun = match.group("noun")
            form = self._genitive_plural(noun)
            if not form or form == noun:
                continue
            out.append(EditCandidate(
                noun, form, 0.997, "rule-quantifier",
                "существительное после «нескольких» в родительном множественного числе",
                start=match.start("noun"),
            ))
        return out

    def _year_phrase(self, text: str) -> list[EditCandidate]:
        if not self.available:
            return []
        out: list[EditCandidate] = []
        for match in YEAR_RE.finditer(text):
            year, adjective, noun = match.group(1), match.group(2), match.group(3)
            try:
                if not 1900 <= int(year) <= 2100:
                    continue
            except ValueError:
                continue
            adj = self.morph_helper.attributive_parses(adjective)
            nouns = self.morph_helper.noun_parses(noun)
            if not adj or not nouns:
                continue
            year_noun = next((p for p in nouns if p.normal_form == "год"), None)
            if year_noun is None:
                continue
            target = {"nomn", "sing", "masc"}
            fixed_adj = None
            for p in adj[:5]:
                candidate = p.inflect(target)
                if candidate and candidate.word != adjective:
                    fixed_adj = preserve_yo(adjective, candidate.word)
                    break
            if not fixed_adj:
                continue
            fixed_adj = preserve_capitalization(adjective, fixed_adj)
            before = f"{year} {adjective} {noun}"
            after = f"{year} {fixed_adj} год"
            if before == after:
                continue
            out.append(EditCandidate(
                before, after, 0.995, "rule-year",
                "конструкция года после числительного", start=match.start(),
            ))
        return out

    # ------------------------------------------------------------------
    # Согласование определения с вершиной
    # ------------------------------------------------------------------
    def _modifier_noun(self, text: str) -> list[EditCandidate]:
        """Согласование в именной группе через унификацию признаков.

        Реализация вынесена в `np_agreement`: там строится модель
        ограничений ИГ (род/одушевлённость вершины, управление предлога,
        краткая форма сказуемого) и правка предлагается только при
        единственном присваивании с максимальной поддержкой.
        """
        if not self.available or not self.agreement_enabled:
            return []
        out: list[EditCandidate] = []
        for start, _end, before, after, reason in self.np_agreement.detect(text):
            out.append(EditCandidate(
                before, after, 0.985, "rule-agreement", reason, start=start,
            ))
        return out

    # ------------------------------------------------------------------
    # Согласование сказуемого с подлежащим
    # ------------------------------------------------------------------
    def _predicative_agreement(self, text: str) -> list[EditCandidate]:
        """«принято меры» → «приняты меры».

        Краткое страдательное причастие согласуется с подлежащим в числе
        и роде. Подлежащее ищется **только справа**: при обратном порядке
        слов («меры принято») ошибка встречается несопоставимо реже, а
        поиск влево даёт ложные срабатывания на генитивных группах
        («Решение о проведении проверки принято руководителем»).

        Выключающие признаки:

        * отрицание («нарушений не выявлено» — безличная конструкция);
        * количественное слово или числительное перед существительным
          («выявлено пять нарушений» — управление родительным);
        * однородный ряд подлежащих («проверены готовность и
          оснащённость» — сказуемое во множественном числе корректно);
        * у существительного нет формы именительного падежа.
        """
        if not self.available or not self.predicative_enabled:
            return []
        matches = list(WORD_RE.finditer(text))
        out: list[EditCandidate] = []
        for idx, match in enumerate(matches):
            word = match.group(0)
            parses = self.morph_helper.known_parses(word)
            short = [p for p in parses if p.tag.POS in {"PRTS", "ADJS"}]
            # Слово должно быть однозначной краткой формой: «принято»
            # разбирается и как PRTS, и как ADJS, но оба разбора дают
            # одно и то же число и род, поэтому вывод не зависит от
            # выбора разбора.
            if not short or len(short) != len(parses):
                continue
            if not any(p.tag.POS == "PRTS" for p in short):
                continue
            if len({(p.tag.number, p.tag.gender) for p in short}) != 1:
                continue
            if idx > 0 and matches[idx - 1].group(0).casefold() in NEGATIONS:
                continue
            subject = self._subject_to_the_right(text, matches, idx)
            if subject is None:
                continue
            subject_idx, subject_parses = subject
            if self._coordinated_subject(text, matches, subject_idx):
                continue
            if not any(p.tag.case == "nomn" for p in subject_parses):
                continue
            nominative = [p for p in subject_parses if p.tag.case == "nomn"]
            if any(
                s.tag.number == n.tag.number
                and (n.tag.number == "plur" or s.tag.gender == n.tag.gender)
                for s in short for n in subject_parses
            ):
                continue
            target = features(nominative[0])
            grammemes = {target.number}
            if target.number != "plur" and target.gender:
                grammemes.add(target.gender)
            produced = {
                form.word
                for form in (p.inflect(grammemes) for p in short[:4])
                if form and form.word
            }
            produced = {w for w in produced if w.lower() != word.lower()}
            if len(produced) != 1:
                continue
            fixed = preserve_capitalization(word, preserve_yo(word, produced.pop()))
            if fixed == word:
                continue
            out.append(EditCandidate(
                word, fixed, 0.975, "rule-predicative",
                f"согласование сказуемого с подлежащим «{matches[subject_idx].group(0)}»",
                start=match.start(),
            ))
        return out

    def _subject_to_the_right(self, text: str, matches: list[re.Match[str]],
                              predicate_idx: int) -> tuple[int, list] | None:
        probe = predicate_idx + 1
        while probe < len(matches):
            gap = text[matches[probe - 1].end():matches[probe].start()]
            if any(ch in CLAUSE_BREAK_CHARS for ch in gap):
                return None
            word = matches[probe].group(0)
            if word.casefold() in GENITIVE_QUANTIFIERS or word.casefold() in NEGATIONS:
                return None
            nouns = self.morph_helper.noun_parses(word)
            if nouns and not self.morph_helper.has_function_reading(word):
                return probe, nouns
            if not self.morph_helper.attributive_parses(word):
                return None
            probe += 1
        return None

    def _coordinated_subject(self, text: str, matches: list[re.Match[str]], subject_idx: int) -> bool:
        for probe in (subject_idx + 1, subject_idx + 2):
            if probe >= len(matches):
                return False
            gap = text[matches[probe - 1].end():matches[probe].start()]
            if any(ch in CLAUSE_BREAK_CHARS for ch in gap):
                return False
            if matches[probe].group(0).casefold() in COORDINATORS:
                return True
        return False

    # ------------------------------------------------------------------
    # Мелкие пунктуационные и управленческие правила
    # ------------------------------------------------------------------
    @staticmethod
    def _po_one(text: str) -> list[EditCandidate]:
        out = []
        for m in PO_ONE_RE.finditer(text):
            before = m.group(0)
            after = re.sub(r"одна$", "одной", before, flags=re.IGNORECASE)
            out.append(EditCandidate(
                before, after, 0.995, "rule-government",
                "форма после предлога «по»", start=m.start(),
            ))
        return out

    def _requires_comma(self, text: str) -> list[EditCandidate]:
        if not self.available:
            return []
        out: list[EditCandidate] = []
        for m in REQUIRES_COMMA_RE.finditer(text):
            word = m.group("word")
            if any(p.tag.case == "gent" for p in self.morph_helper.noun_parses(word)):
                out.append(EditCandidate(
                    "требует, ", "требует ", 0.975, "rule-punctuation",
                    "запятая между сказуемым и генитивным дополнением",
                    start=m.start(),
                ))
        return out

    def candidates(self, text: str) -> list[EditCandidate]:
        out: list[EditCandidate] = []
        seen: set[tuple[str, str, str, int | None]] = set()
        for group in (
            self._quantifier_government(text),
            self._year_phrase(text),
            self._modifier_noun(text),
            self._predicative_agreement(text),
            self._po_one(text),
            self._requires_comma(text),
        ):
            for c in group:
                key = (c.before, c.after, c.category, c.start)
                if key not in seen:
                    seen.add(key)
                    out.append(c)
        return out
