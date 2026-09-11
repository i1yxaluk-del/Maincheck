"""Проверка кандидатов и голосование между источниками (v9).

Две задачи модуля
=================

**1. `GenerativeGuard` — фильтр галлюцинаций генеративных стадий.**

В v8 защита `DecisionEngine._is_unverified_llm_inflection` срабатывала
только для категорий `model*`, `unknown*`, `languagetool*`, `surface*`.
Реальные же категории v8-стека называются `sage-spell-punc`,
`russian-gec`, `draft_tlite`, `draft_giga` — ни одна не попадала под
префиксы, т.е. антигаллюцинационный фильтр в проде был мёртвым кодом.
Любая падежная фантазия SAGE проходила с уверенностью 0.90.

Guard закрывает это по классам правок, а не по именам категорий:

* падежная правка уже согласованного в контексте слова — отклоняется;
* лексическая замена словарного слова (другая лемма) — отклоняется,
  т.к. модель не может доказать, что автор имел в виду другое слово;
* правка, меняющая цифры, даты, номера пунктов — отклоняется;
* правка внутри инициалов/сокращений («п. 3», «ст. 12», «т.д.») —
  отклоняется;
* правки пунктуации и опечаток (слово вне словаря) — разрешаются: это
  то, для чего SAGE и GEC действительно пригодны.

**2. `CandidateArbiter` — голосование вместо «кто первый, того и правка».**

v8 принимал правку от одной слабой стадии, если её confidence выше
порога. v9 требует подтверждения: инфлективная правка, предложенная
единственной генеративной моделью и не подтверждённая ни детерминированным
правилом, ни второй моделью, понижается в уверенности ниже порога
принятия. Это классический system-combination приём из GEC, и он даёт
основной прирост точности без потери детерминированной полноты.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, replace

from decision_engine import EditCandidate
from morphology import Morphology, features, get_morphology, yo_equal

log = logging.getLogger("ai_suggester.verification")

#: Источники, чьи правки морфологически доказаны на стороне сервера.
DETERMINISTIC_PREFIXES = ("rule-", "dict-spell", "languagetool-")

#: Источники, порождающие текст целиком; их правки требуют проверки.
GENERATIVE_PREFIXES = ("sage", "russian-gec", "draft", "model", "surface", "unknown", "diff")

WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z]+")
#: «Скелет» фрагмента: слова и числа. Пунктуация не входит, поэтому
#: правки, отличающиеся только знаками, распознаются как безопасный класс.
SKELETON_RE = re.compile(r"[А-Яа-яЁёA-Za-z]+|\d+")
DIGIT_RE = re.compile(r"\d")
ABBREV_RE = re.compile(r"\b(?:[а-яё]{1,4}\.|[А-ЯЁ]\.)")
PUNCT_ONLY = re.compile(r"^[\s.,;:!?()\[\]«»\"'—–-]*$")


def is_generative(category: str) -> bool:
    return category.startswith(GENERATIVE_PREFIXES)


def is_deterministic(category: str) -> bool:
    return category.startswith(DETERMINISTIC_PREFIXES)


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    reason: str = ""


class GenerativeGuard:
    """Классовый фильтр правок, приходящих от генеративных стадий."""

    def __init__(self, morphology: Morphology | None = None,
                 protected_words: set[str] | None = None) -> None:
        self.morph = morphology or get_morphology()
        self.protected = {w.casefold() for w in (protected_words or set())}
        self.enabled = os.getenv("GENERATIVE_GUARD_ENABLED", "true").lower() in {
            "1", "true", "yes", "on",
        }
        self._rejected: dict[str, int] = {}

    # ------------------------------------------------------------------
    @property
    def rejections(self) -> dict[str, int]:
        return dict(self._rejected)

    def _reject(self, reason: str) -> tuple[bool, str]:
        self._rejected[reason] = self._rejected.get(reason, 0) + 1
        return False, reason

    def allow(self, candidate: EditCandidate, text: str) -> tuple[bool, str]:
        if not self.enabled:
            return True, ""
        before, after = candidate.before, candidate.after
        if not is_generative(candidate.category):
            return True, ""

        if self._changes_digits(before, after):
            return self._reject("правка меняет числа или даты")

        # Пунктуация и пробелы — разрешённый класс: сравниваем «скелет»
        # из слов и чисел.
        if SKELETON_RE.findall(before) == SKELETON_RE.findall(after):
            return True, ""

        if self._touches_abbreviation(before, after):
            return self._reject("правка внутри сокращения или инициалов")

        word_pairs = self._word_pairs(before, after)
        if word_pairs is None:
            # Изменено количество слов вне пунктуационного класса — модель
            # переписывает фрагмент, а не исправляет ошибку.
            return self._reject("изменено количество слов")

        for src, dst in word_pairs:
            allowed, reason = self._check_word_pair(src, dst, text)
            if not allowed:
                return self._reject(reason)
        return True, ""

    # ------------------------------------------------------------------
    @staticmethod
    def _changes_digits(before: str, after: str) -> bool:
        return DIGIT_RE.findall(before) != DIGIT_RE.findall(after)

    @staticmethod
    def _touches_abbreviation(before: str, after: str) -> bool:
        return bool(ABBREV_RE.search(before)) and ABBREV_RE.findall(before) != ABBREV_RE.findall(after)

    @staticmethod
    def _word_pairs(before: str, after: str) -> list[tuple[str, str]] | None:
        src_words = WORD_RE.findall(before)
        dst_words = WORD_RE.findall(after)
        if len(src_words) != len(dst_words):
            return None
        return [(s, d) for s, d in zip(src_words, dst_words) if s != d]

    def _check_word_pair(self, src: str, dst: str, text: str) -> tuple[bool, str]:
        if src.casefold() in self.protected:
            return False, "защищённый термин"
        if yo_equal(src, dst):
            return False, "различие только в ё/е"
        if not self.morph.available:
            return True, ""

        # Слова нет в словаре — это опечатка, генеративная правка уместна.
        if not self.morph.is_known(src):
            return True, ""

        src_lemmas = self.morph.lemmas(src)
        dst_lemmas = self.morph.lemmas(dst)

        # Другая лемма при словарном исходном слове: модель подменяет
        # лексику («раздел» → «разделение»), доказать это она не может.
        if src_lemmas and dst_lemmas and not (src_lemmas & dst_lemmas):
            return False, "лексическая подмена словарного слова"

        # Та же лемма: инфлективная правка. Разрешаем только если исходная
        # форма действительно рассогласована с ближайшей вершиной/определением.
        if src_lemmas & dst_lemmas:
            if self._agrees_in_context(src, text):
                return False, "падежная правка уже согласованной формы"
        return True, ""

    def _agrees_in_context(self, word: str, text: str) -> bool:
        """True, если слово согласовано с соседом в тексте.

        Проверяются обе стороны: слово как определение при следующем
        существительном и как вершина при предыдущем определении. Если
        хотя бы одна связь согласована — форма в тексте валидна, и
        генеративная правка падежа является «улучшением», а не
        исправлением.
        """
        matches = list(WORD_RE.finditer(text))
        index = next((i for i, m in enumerate(matches) if m.group(0) == word), None)
        if index is None:
            return False
        neighbours: list[tuple[str, str]] = []
        if index + 1 < len(matches):
            neighbours.append(("modifier", matches[index + 1].group(0)))
        if index > 0:
            neighbours.append(("head", matches[index - 1].group(0)))

        for role, neighbour in neighbours:
            if role == "modifier":
                modifier, head = word, neighbour
            else:
                modifier, head = neighbour, word
            if not self.morph.attributive_parses(modifier):
                continue
            if not self.morph.noun_parses(head):
                continue
            if self.morph.has_function_reading(head):
                continue
            if self.morph.pair_agrees(modifier, head):
                return True
        return False


class CandidateArbiter:
    """Слияние кандидатов от разных стадий с голосованием."""

    #: Базовое доверие к источнику. Детерминированные стадии выше любой
    #: модели. Калибровка привязана к `DECISION_MIN_CONFIDENCE=0.55` и
    #: `ARBITER_SOLO_PENALTY=0.25`:
    #:
    #:   russian-gec  0.86 − 0.25 = 0.61  → одиночная правка принимается
    #:   sage         0.80 − 0.25 = 0.55  → принимается на границе
    #:   draft_*      0.66 − 0.25 = 0.41  → требуется подтверждение
    #:
    #: Смысл: специализированный GEC-корректор — целевая модель для этого
    #: класса ошибок, и её одиночному голосу мы доверяем (галлюцинации у
    #: неё отсекает `GenerativeGuard`). Универсальные rescue-генераторы
    #: перепишут что угодно, поэтому их инфлективные правки без второго
    #: голоса не применяются.
    WEIGHTS = {
        "rule-": 1.00,
        "dict-spell": 0.94,
        "languagetool-": 0.90,
        "russian-gec": 0.86,
        "sage": 0.80,
        "draft": 0.66,
    }

    def __init__(self) -> None:
        self.min_votes_for_inflection = int(os.getenv("ARBITER_MIN_VOTES_INFLECTION", "2"))
        self.corroboration_bonus = float(os.getenv("ARBITER_CORROBORATION_BONUS", "0.10"))
        self.solo_generative_penalty = float(os.getenv("ARBITER_SOLO_PENALTY", "0.25"))
        self.morph = get_morphology()

    def weight(self, category: str) -> float:
        for prefix, value in self.WEIGHTS.items():
            if category.startswith(prefix):
                return value
        return 0.5

    def _is_inflection(self, candidate: EditCandidate) -> bool:
        if not self.morph.available:
            return False
        src = WORD_RE.findall(candidate.before)
        dst = WORD_RE.findall(candidate.after)
        if len(src) != len(dst):
            return False
        for s, d in zip(src, dst):
            if s == d:
                continue
            if self.morph.lemmas(s) & self.morph.lemmas(d):
                return True
        return False

    def merge(self, candidates: list[EditCandidate]) -> list[EditCandidate]:
        """Схлопывает одинаковые правки, суммируя голоса источников."""
        buckets: dict[tuple[int | None, str, str], list[EditCandidate]] = {}
        for candidate in candidates:
            key = (candidate.start, candidate.before, candidate.after)
            buckets.setdefault(key, []).append(candidate)

        merged: list[EditCandidate] = []
        for (start, before, after), group in buckets.items():
            best = max(group, key=lambda c: self.weight(c.category))
            sources = tuple(sorted({c.category for c in group}))
            score = self.weight(best.category)
            if len(sources) > 1:
                score = min(1.0, score + self.corroboration_bonus * (len(sources) - 1))
            has_deterministic = any(is_deterministic(c.category) for c in group)
            if (
                not has_deterministic
                and len(sources) < self.min_votes_for_inflection
                and self._is_inflection(best)
            ):
                # Инфлективная правка от одной модели без подтверждения.
                score -= self.solo_generative_penalty
            merged.append(replace(
                best,
                confidence=round(max(0.0, min(1.0, score)), 4),
                sources=sources,
                start=start,
            ))
        merged.sort(key=lambda c: (c.confidence, len(c.before)), reverse=True)
        return merged
