"""Согласование внутри именной группы через унификацию признаков (v9).

Почему не «прилагательное рядом с существительным»
==================================================
Подход v8 («взять лучший разбор вершины, подогнать определение») давал
нулевую точность на официально-деловом тексте: русская морфология
омонимична, и выбор одного разбора превращает корректную фразу в
«ошибку». Эталонный пример из прод-отчёта — «в соответствующий раздел»
→ «в соответствующего раздел».

Модель v9
=========
Именная группа (ИГ) рассматривается как набор токенов с множествами
допустимых прочтений `(падеж, число, род, одушевлённость)`. Задача —
найти *присваивание* признаков, согласованное с максимальным числом
токенов (унификация). Правка предлагается только тогда, когда:

* найдено **единственное** присваивание с максимальной поддержкой
  (либо все присваивания-лидеры дают одну и ту же правку);
* ровно один токен ИГ с ним несовместим;
* для этого токена существует словарная форма, удовлетворяющая
  присваиванию.

Дополнительные источники ограничений
====================================
1. **Род и одушевлённость вершины** — лексические факты, не омонимия.
   Они сужают прочтения определений (и снимают дефект
   `inflect({accs, sing, masc})` → одушевлённая форма).
2. **Управление предлога** — только однозначные предлоги («согласно» →
   датив, «при» → предложный, «для» → родительный). Многозначные («в»,
   «на», «по», «с», «за», «под», «между») исключены.
3. **Краткая форма сказуемого** (PRTS/ADJS) — подлежащее пассивной
   конструкции стоит в именительном падеже, что снимает омонимию
   «Проверочное мероприятия проведено».

Что исключается из ИГ (закрытые классы ложных срабатываний)
===========================================================
* причастный оборот, примыкающий влево («ущерб, причинённый
  подразделению» — «причинённый» согласуется с «ущерб»);
* предикативный творительный («назначенный ответственным сотрудник»,
  «признан виновным»);
* субстантивированные местоимения («при этом количество…»);
* омонимичные существительные в позиции определения («с данными
  бухгалтерского учёта»);
* однородный ряд определений при вершине во множественном числе
  («для первого и третьего отделов») — число не правим.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable

from morphology import (
    Morphology,
    features,
    get_morphology,
    preserve_capitalization,
    preserve_yo,
)

log = logging.getLogger("ai_suggester.np_agreement")

WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z]+(?:[-/][А-Яа-яЁёA-Za-z]+)*")

#: Предлоги с однозначным управлением. Многозначные («в», «на», «по»,
#: «с», «за», «под», «между») сознательно исключены: они дают не
#: ограничение, а ложные срабатывания.
PREP_GOVERNMENT: dict[str, frozenset[str]] = {
    # датив
    "согласно": frozenset({"datv"}),
    "благодаря": frozenset({"datv"}),
    "вопреки": frozenset({"datv"}),
    "наперекор": frozenset({"datv"}),
    "подобно": frozenset({"datv"}),
    "соразмерно": frozenset({"datv"}),
    "к": frozenset({"datv"}),
    "ко": frozenset({"datv"}),
    # генитив
    "без": frozenset({"gent"}),
    "для": frozenset({"gent"}),
    "до": frozenset({"gent"}),
    "из": frozenset({"gent"}),
    "из-за": frozenset({"gent"}),
    "из-под": frozenset({"gent"}),
    "от": frozenset({"gent"}),
    "у": frozenset({"gent"}),
    "кроме": frozenset({"gent"}),
    "вместо": frozenset({"gent"}),
    "вокруг": frozenset({"gent"}),
    "возле": frozenset({"gent"}),
    "около": frozenset({"gent"}),
    "после": frozenset({"gent"}),
    "против": frozenset({"gent"}),
    "среди": frozenset({"gent"}),
    "путем": frozenset({"gent"}),
    "путём": frozenset({"gent"}),
    "посредством": frozenset({"gent"}),
    "относительно": frozenset({"gent"}),
    "касательно": frozenset({"gent"}),
    "ввиду": frozenset({"gent"}),
    "вследствие": frozenset({"gent"}),
    "сверх": frozenset({"gent"}),
    "мимо": frozenset({"gent"}),
    "внутри": frozenset({"gent"}),
    "вне": frozenset({"gent"}),
    # аккузатив
    "через": frozenset({"accs"}),
    "сквозь": frozenset({"accs"}),
    "про": frozenset({"accs"}),
    # инструментатив
    "над": frozenset({"ablt"}),
    "перед": frozenset({"ablt"}),
    "пред": frozenset({"ablt"}),
    # локатив
    "при": frozenset({"loct"}),
    "о": frozenset({"loct"}),
    "об": frozenset({"loct"}),
    "обо": frozenset({"loct"}),
}

#: Начала составных предлогов: «в течение», «в ходе», «в целях»,
#: «в соответствии», «по результатам». Если предлог однозначный, но за
#: ним стоит такое слово, ограничение не применяется.
COORDINATORS = frozenset({"и", "или", "либо", "а", "но", "также"})

#: Квантификаторы, управляющие родительным падежом: при них согласование
#: «определение + вершина» не проверяется.
GENITIVE_QUANTIFIERS = frozenset({
    "несколько", "нескольких", "много", "немного", "ряд", "ряда", "большинство",
    "меньшинство", "часть", "количество", "число", "масса", "множество", "сколько",
    "столько", "мало", "немало", "свыше", "более", "менее", "около",
})

CLAUSE_BREAK = frozenset(",;:.!?()[]{}«»\"„“”—–\n")

NEGATIONS = frozenset({"не", "ни"})


@dataclass(frozen=True)
class Reading:
    case: str | None
    number: str | None
    gender: str | None
    animacy: str | None = None

    def key(self) -> tuple[str | None, str | None, str | None]:
        """Признаки, по которым проверяется согласование."""
        gender = None if self.number == "plur" else self.gender
        return (self.case, self.number, gender)


@dataclass
class Token:
    index: int
    text: str
    start: int
    end: int
    parses: list[Any]
    readings: set[Reading]
    is_head: bool = False


@dataclass
class Assignment:
    case: str
    number: str
    gender: str | None
    animacy: str | None
    support: float = 0.0
    supporters: tuple[int, ...] = ()


def _readings_of(parses: Iterable[Any]) -> set[Reading]:
    out: set[Reading] = set()
    for parse in parses:
        f = features(parse)
        if not f.case or not f.number:
            continue
        out.add(Reading(f.case, f.number, f.gender, f.animacy))
    return out


class NounPhraseAgreement:
    """Детектор рассогласования внутри именной группы."""

    def __init__(self, morphology: Morphology | None = None) -> None:
        self.morph = morphology or get_morphology()

    @property
    def available(self) -> bool:
        return self.morph.available

    # ------------------------------------------------------------------
    # Разбор текста на именные группы
    # ------------------------------------------------------------------
    def _attributive(self, word: str) -> list[Any]:
        """Определения ИГ: полные прилагательные, причастия и «один».

        Числительное «один» — единственное, которое согласуется с
        существительным; «два», «три», «пять» управляют родительным
        падежом, поэтому в ИГ как определения не входят.
        """
        parses = list(self.morph.attributive_parses(word))
        parses += [
            p for p in self.morph.known_parses(word)
            if p.tag.POS == "NUMR" and p.normal_form == "один"
        ]
        return parses

    def _can_head(self, word: str) -> bool:
        """True, если слово может быть вершиной именной группы.

        Отсекаются местоимённые прилагательные («той», «том», «этом»):
        у них в словаре OpenCorpora есть и NOUN-разборы, из-за чего
        «по одной и той же теме» разбиралось как ИГ с вершиной «той».
        Кроме того, требуем, чтобы существительное было не менее
        вероятным разбором, чем прилагательное или причастие.
        """
        noun = self.morph.noun_parses(word)
        if not noun:
            return False
        attributive = self.morph.attributive_parses(word)
        if any("Apro" in str(p.tag) for p in attributive):
            return False
        if attributive and max(p.score for p in attributive) > max(p.score for p in noun):
            return False
        return True

    def _substantivizable_pronoun(self, word: str) -> bool:
        """True для местоимений, способных стоять при предлоге одиночно.

        «при этом», «в том», «о тех» — местоимение здесь объект предлога,
        а не определение следующего существительного. Числительные-
        местоимения («один», «одно» — грамммема `Anum`) так не ведут
        себя и остаются определениями.
        """
        parses = self.morph.attributive_parses(word)
        if not parses:
            return False
        return any("Apro" in str(p.tag) and "Anum" not in str(p.tag) for p in parses)

    def _preceded_by_governing_prep(self, text: str, matches: list[re.Match[str]], idx: int) -> bool:
        if idx == 0:
            return False
        previous = matches[idx - 1].group(0).casefold()
        if previous not in PREP_GOVERNMENT:
            return False
        return not self._gap_breaks_clause(text, matches[idx - 1].end(), matches[idx].start())

    def _gap_breaks_clause(self, text: str, left_end: int, right_start: int) -> bool:
        return any(ch in CLAUSE_BREAK for ch in text[left_end:right_start])

    def _attaches_left(self, text: str, matches: list[re.Match[str]], idx: int) -> bool:
        """True, если полное причастие относится к слову слева.

        Два независимых признака обособленного причастного оборота:
        запятая непосредственно перед причастием и согласование с
        ближайшим существительным слева.
        """
        word = matches[idx].group(0)
        parses = self.morph.attributive_parses(word)
        if not parses or not all(p.tag.POS == "PRTF" for p in parses):
            return False
        prefix = text[:matches[idx].start()].rstrip()
        if prefix.endswith(","):
            return True
        for j in range(idx - 1, max(-1, idx - 4), -1):
            candidate = matches[j].group(0)
            if self.morph.has_function_reading(candidate):
                continue
            if not self.morph.noun_parses(candidate):
                continue
            return self.morph.pair_agrees(word, candidate)
        return False

    def _predicative_instrumental(self, word: str, head_word: str) -> bool:
        """True для предикативного творительного.

        «назначенный ответственным сотрудник», «признан виновным»,
        «считается недействительным»: слово в творительном падеже при
        вершине, которая творительный принимать не может, — это не
        определение, а именная часть сказуемого.

        Сравниваем только прочтения, совпадающие с вершиной по числу и
        роду: у прилагательных женского рода формы родительного,
        дательного, творительного и предложного падежей совпадают
        («профессиональной»), и проверка «есть ли ablt-прочтение» без
        этого сужения запрещала бы правку целого класса ошибок.
        """
        readings = _readings_of(self._attributive(word))
        if not any(r.case == "ablt" for r in readings):
            return False
        head_readings = _readings_of(self.morph.noun_parses(head_word))
        if not head_readings or any(r.case == "ablt" for r in head_readings):
            return False
        comparable = [
            r for r in readings
            if any(
                r.number == h.number and (r.number == "plur" or r.gender == h.gender)
                for h in head_readings
            )
        ]
        return bool(comparable) and all(r.case == "ablt" for r in comparable)

    def _collect(self, text: str, matches: list[re.Match[str]],
                 head_idx: int) -> tuple[list[int], str | None, bool, bool]:
        """Собирает определения слева от вершины.

        Возвращает (индексы определений, предлог, был ли однородный ряд,
        признак именного управления слева).
        """
        modifiers: list[int] = []
        coordinated = False
        genitive_hint = False
        right = matches[head_idx].start()
        j = head_idx - 1
        head_word = matches[head_idx].group(0)

        while j >= 0:
            word = matches[j].group(0)
            if self._gap_breaks_clause(text, matches[j].end(), right):
                break
            lowered = word.casefold()
            if lowered in COORDINATORS:
                coordinated = True
                right = matches[j].start()
                j -= 1
                continue
            attributive = self._attributive(word)
            if not attributive:
                break
            # Омонимичное существительное в позиции определения
            # («с данными бухгалтерского учёта») — граница ИГ.
            if self.morph.noun_parses(word) and not self.morph.pair_agrees(word, head_word):
                break
            # Субстантивированное местоимение («при этом количество…»).
            if (
                self._substantivizable_pronoun(word)
                and not self.morph.pair_agrees(word, head_word)
                and self._preceded_by_governing_prep(text, matches, j)
            ):
                return modifiers, None, coordinated, False
            if self._attaches_left(text, matches, j):
                break
            if self._predicative_instrumental(word, head_word):
                break
            modifiers.append(j)
            right = matches[j].start()
            j -= 1

        prep = None
        if j >= 0 and not self._gap_breaks_clause(text, matches[j].end(), right):
            candidate = matches[j].group(0)
            lowered = candidate.casefold()
            if lowered in PREP_GOVERNMENT and self.morph.has_function_reading(lowered):
                prep = lowered
            elif (
                not self.morph.has_function_reading(candidate)
                and self.morph.noun_parses(candidate)
                and lowered not in GENITIVE_QUANTIFIERS
            ):
                # Именное управление «факт (чего?) оформления»: вершина
                # такой группы почти всегда в родительном падеже.
                genitive_hint = True
        modifiers.reverse()
        return modifiers, prep, coordinated, genitive_hint

    def _predicative(self, text: str, matches: list[re.Match[str]],
                     first_idx: int, head_idx: int) -> Any | None:
        """Краткая форма сказуемого, согласованная с подлежащим ИГ.

        Ищем непосредственно перед ИГ и непосредственно после вершины,
        не выходя за границу предложения. Отрицание и генитивные
        квантификаторы отключают признак: «нарушений не выявлено» —
        безличная конструкция, а не рассогласование.
        """
        def short_form(word: str) -> Any | None:
            parses = self.morph.known_parses(word)
            if not parses:
                return None
            # Требуем однозначную краткую форму. «согласно» тоже имеет
            # ADJS-разбор, но это предлог, и подлежащего у него нет.
            if any(p.tag.POS not in {"PRTS", "ADJS"} for p in parses):
                return None
            if len({(features(p).number, features(p).gender) for p in parses}) != 1:
                return None
            return parses[0]

        for probe, boundary_left, boundary_right in (
            (first_idx - 1, first_idx - 1, first_idx),
            (head_idx + 1, head_idx, head_idx + 1),
        ):
            if probe < 0 or probe >= len(matches):
                continue
            left = min(boundary_left, boundary_right)
            if self._gap_breaks_clause(text, matches[left].end(), matches[left + 1].start()):
                continue
            word = matches[probe].group(0)
            if word.casefold() in NEGATIONS:
                continue
            if probe > 0 and matches[probe - 1].group(0).casefold() in NEGATIONS:
                continue
            parse = short_form(word)
            if parse is not None:
                return parse
        return None

    def _predicative_fits_head(self, predicative: Any, head_parses: list[Any]) -> bool:
        """Краткая форма должна быть совместима с вершиной ИГ.

        В «Согласно утверждённому плана проведена штабная тренировка»
        сказуемое «проведена» относится к «тренировка», а не к «плана»:
        без этой проверки оно давало ложное ограничение и блокировало
        правку управления предлога.
        """
        pf = features(predicative)
        for parse in head_parses:
            hf = features(parse)
            if hf.number != pf.number:
                continue
            if pf.number != "plur" and hf.gender and pf.gender and hf.gender != pf.gender:
                continue
            return True
        return False

    # ------------------------------------------------------------------
    # Унификация
    # ------------------------------------------------------------------
    def _candidates_for_np(self, text: str, matches: list[re.Match[str]],
                           head_idx: int) -> list[tuple[Token, Assignment]]:
        head_word = matches[head_idx].group(0)
        head_parses = self.morph.noun_parses(head_word)
        if not head_parses:
            return []

        modifier_indexes, prep, coordinated, genitive_hint = self._collect(text, matches, head_idx)

        head_genders = {features(p).gender for p in head_parses if features(p).gender}
        head_animacy = {features(p).animacy for p in head_parses if features(p).animacy}

        def lexically_allowed(reading: Reading, token_is_head: bool) -> bool:
            if token_is_head:
                return True
            if head_genders and reading.number != "plur" and reading.gender and reading.gender not in head_genders:
                return False
            if head_animacy and reading.animacy and reading.animacy not in head_animacy:
                return False
            return True

        tokens: list[Token] = []
        for idx in modifier_indexes + [head_idx]:
            match = matches[idx]
            word = match.group(0)
            parses = (
                self.morph.noun_parses(word) if idx == head_idx
                else self._attributive(word)
            )
            raw = _readings_of(parses)
            if idx != head_idx and not raw:
                continue
            allowed = {r for r in raw if lexically_allowed(r, idx == head_idx)}
            if prep:
                cases = PREP_GOVERNMENT[prep]
                allowed = {r for r in allowed if r.case in cases}
            tokens.append(Token(idx, word, match.start(), match.end(), parses, allowed,
                                is_head=idx == head_idx))

        head_token = tokens[-1]
        modifiers = tokens[:-1]

        # Предикативный корроборатор используется только при наличии
        # определений: одиночное существительное по сказуемому не правим.
        # `genitive_hint` означает, что ИГ — зависимая генитивная группа
        # при предыдущем существительном («Результаты контрольной
        # проверки признаны…»). Подлежащим сказуемого тогда является то
        # предыдущее существительное, и краткая форма ничего не говорит
        # о падеже этой ИГ.
        predicative = None
        if modifiers and not genitive_hint:
            first_idx = modifiers[0].index
            if not any(
                m.text.casefold() in GENITIVE_QUANTIFIERS for m in modifiers
            ):
                predicative = self._predicative(text, matches, first_idx, head_idx)
                if predicative is not None and not self._predicative_fits_head(predicative, head_parses):
                    predicative = None

        # --- одиночное существительное после однозначного предлога -----
        if not modifiers:
            if not prep or head_token.readings:
                return []
            required = next(iter(PREP_GOVERNMENT[prep])) if len(PREP_GOVERNMENT[prep]) == 1 else None
            if not required:
                return []
            base = next(iter(_readings_of(head_parses)), None)
            if base is None:
                return []
            assignment = Assignment(required, base.number, base.gender, base.animacy, 1.0, ())
            return [(head_token, assignment)]

        # --- сбор присваиваний ------------------------------------------
        pool: dict[tuple[str | None, str | None, str | None], Assignment] = {}
        for token in tokens:
            for reading in token.readings:
                key = reading.key()
                if key[0] is None or key[1] is None:
                    continue
                pool.setdefault(key, Assignment(
                    reading.case, reading.number, reading.gender,
                    next(iter(head_animacy)) if head_animacy else reading.animacy,
                ))

        if predicative is not None:
            pf = features(predicative)
            key = ("nomn", pf.number, None if pf.number == "plur" else pf.gender)
            pool.setdefault(key, Assignment(
                "nomn", pf.number or "sing", key[2],
                next(iter(head_animacy)) if head_animacy else None,
            ))

        if not pool:
            return []

        for key, assignment in pool.items():
            supporters: list[int] = []
            support = 0.0
            for token in tokens:
                if any(r.key() == key for r in token.readings):
                    support += 1.0
                    supporters.append(token.index)
            if predicative is not None:
                pf = features(predicative)
                predicative_key = ("nomn", pf.number, None if pf.number == "plur" else pf.gender)
                if key == predicative_key:
                    support += 2.0
            if genitive_hint and key[0] == "gent":
                support += 1.0
            assignment.support = support
            assignment.supporters = tuple(supporters)

        best = max(a.support for a in pool.values())
        if best <= 0:
            return []
        leaders = [a for a in pool.values() if a.support == best]

        if len(leaders) > 1:
            # Приоритет присваиванию, поддержанному наименее омонимичным
            # токеном: он несёт больше информации о структуре ИГ.
            ambiguity = {
                token.index: len({r.key() for r in _readings_of(token.parses)})
                for token in tokens
            }
            scored = [
                (min((ambiguity[i] for i in a.supporters), default=99), a)
                for a in leaders
            ]
            lowest = min(score for score, _ in scored)
            leaders = [a for score, a in scored if score == lowest]

        proposals: list[tuple[Token, Assignment]] = []
        for assignment in leaders:
            key = (assignment.case, assignment.number,
                   None if assignment.number == "plur" else assignment.gender)
            inconsistent = [t for t in tokens if not any(r.key() == key for r in t.readings)]
            if len(inconsistent) != 1:
                continue
            token = inconsistent[0]
            if coordinated and not self._same_number(token, assignment):
                continue  # однородный ряд определений при вершине в мн. ч.
            if token.is_head and not self._same_number(token, assignment):
                if predicative is None and not prep:
                    continue  # смена числа вершины требует подтверждения
            proposals.append((token, assignment))

        if not proposals:
            return []
        targets = {token.index for token, _ in proposals}
        if len(targets) != 1:
            return []  # лидеры расходятся в том, какой токен править
        return proposals

    @staticmethod
    def _same_number(token: Token, assignment: Assignment) -> bool:
        return any(r.number == assignment.number for r in _readings_of(token.parses))

    def _apply_assignment(self, token: Token, assignment: Assignment) -> str | None:
        base = {assignment.case, assignment.number}
        if assignment.number != "plur" and assignment.gender:
            base.add(assignment.gender)
        # Одушевлённость различает формы только в винительном падеже;
        # в остальных она не является грамммемой словоформы, и её
        # добавление заставляет `inflect` вернуть None.
        variants = [base | {assignment.animacy}] if (
            assignment.animacy and assignment.case == "accs"
        ) else []
        variants.append(base)
        produced: set[str] = set()
        for grammemes in variants:
            for parse in token.parses[:8]:
                try:
                    form = parse.inflect(grammemes)
                except Exception:  # pragma: no cover
                    form = None
                if form and form.word:
                    produced.add(form.word)
            if produced:
                break
        produced = {w for w in produced if w.lower() != token.text.lower()}
        if len(produced) != 1:
            return None
        word = preserve_capitalization(token.text, preserve_yo(token.text, produced.pop()))
        return word if word != token.text else None

    # ------------------------------------------------------------------
    def detect(self, text: str) -> list[tuple[int, int, str, str, str]]:
        """Возвращает (start, end, before, after, reason) для каждой правки."""
        if not self.available or not text:
            return []
        matches = list(WORD_RE.finditer(text))
        out: list[tuple[int, int, str, str, str]] = []
        seen: set[int] = set()
        for head_idx, match in enumerate(matches):
            head_word = match.group(0)
            if self.morph.has_function_reading(head_word):
                continue
            if not self._can_head(head_word):
                continue
            if head_word.casefold() in GENITIVE_QUANTIFIERS:
                continue
            try:
                proposals = self._candidates_for_np(text, matches, head_idx)
            except Exception as exc:  # pragma: no cover
                log.warning("NP agreement failed on %r: %s", head_word, exc)
                continue
            for token, assignment in proposals[:1]:
                if token.start in seen or "-" in token.text:
                    continue
                fixed = self._apply_assignment(token, assignment)
                if not fixed:
                    continue
                seen.add(token.start)
                role = "вершины" if token.is_head else "определения"
                out.append((
                    token.start, token.end, token.text, fixed,
                    f"согласование {role} в именной группе «{head_word}» "
                    f"({assignment.case}, {assignment.number})",
                ))
        return out
