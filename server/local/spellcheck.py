"""Детерминированная проверка орфографии по морфологическому словарю (v9).

Мотивация
=========
Жалоба «не находит ошибок орфографии» не решается генеративной моделью:
SAGE-distilled-95m обучен на шумных веб-текстах, на официально-деловой
лексике он чаще фантазирует, чем исправляет. Между тем для опечаток
существует детерминированное решение с высокой точностью:

1. слово отсутствует в словаре OpenCorpora (pymorphy3) → это кандидат на
   опечатку;
2. среди правок на расстоянии Дамерау—Левенштейна 1 ищем словарные формы;
3. правку предлагаем только если словарный вариант **единственный**.

Единственность — жёсткий, но правильный критерий: «законости» →
«законности» (единственный вариант) исправляем, а слово с двумя
равновероятными вариантами отдаём моделям и человеку.

Цена: один DAWG-lookup ≈ 19 мкс, на слово из 12 символов ≈ 800 lookup-ов
≈ 15 мс, и только для слов, которых нет в словаре. На типичном фрагменте
документа это единицы миллисекунд.
"""

from __future__ import annotations

import logging
import os
import re

from decision_engine import EditCandidate
from morphology import Morphology, get_morphology

log = logging.getLogger("ai_suggester.spellcheck")

ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
TOKEN_RE = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*")
SENTENCE_START_RE = re.compile(r"(?:^|[.!?;:]\s+|\n\s*)$")
CLAUSE_BREAK = frozenset(",;:.!?()[]{}«»\"„“”—–\n")

#: Стилистические и вариантные пометы OpenCorpora. Словоформа, у которой
#: **все** разборы помечены так, является нестандартной («подразделенья»
#: с пометой `V-be`, «Erro» — заведомо ошибочная форма) и не может быть
#: результатом исправления опечатки.
NONSTANDARD_MARKS = (
    "Arch", "Litr", "Erro", "Dist", "Infr", "Slng", "Ques", "Prnt",
    "V-be", "V-en", "V-ie", "V-bi", "V-sh", "V-oy", "V-ey",
)


class DictionarySpellChecker:
    """Опечатки, доказуемые словарём, без обращения к моделям."""

    def __init__(self, morphology: Morphology | None = None,
                 protected_words: set[str] | None = None) -> None:
        self.morph = morphology or get_morphology()
        self.enabled = os.getenv("DICT_SPELLCHECK_ENABLED", "true").lower() in {
            "1", "true", "yes", "on",
        }
        self.min_length = int(os.getenv("DICT_SPELLCHECK_MIN_LENGTH", "5"))
        self.max_length = int(os.getenv("DICT_SPELLCHECK_MAX_LENGTH", "24"))
        self.confidence = float(os.getenv("DICT_SPELLCHECK_CONFIDENCE", "0.93"))
        self.protected = {w.casefold() for w in (protected_words or set())}

    @property
    def available(self) -> bool:
        return self.enabled and self.morph.available

    def _known(self, word: str) -> bool:
        analyzer = self.morph._morph
        if analyzer is None:
            return True
        try:
            return bool(analyzer.word_is_known(word))
        except Exception:  # pragma: no cover
            return True

    def _edits1(self, word: str) -> set[str]:
        """Правки на расстоянии Дамерау—Левенштейна 1 в русском алфавите."""
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = {left + right[1:] for left, right in splits if right}
        transposes = {
            left + right[1] + right[0] + right[2:]
            for left, right in splits if len(right) > 1
        }
        replaces = {
            left + letter + right[1:]
            for left, right in splits if right for letter in ALPHABET
        }
        inserts = {left + letter + right for left, right in splits for letter in ALPHABET}
        return (deletes | transposes | replaces | inserts) - {word}

    def _is_standard_form(self, word: str) -> bool:
        parses = self.morph.known_parses(word)
        if not parses:
            return False
        return any(
            not any(mark in str(p.tag) for mark in NONSTANDARD_MARKS) for p in parses
        )

    def _agrees_with_context(self, text: str, match: re.Match[str], variant: str) -> bool:
        """Проверяет вариант по согласованию с соседями по именной группе.

        «знаний нормативых правовых актов» → из вариантов «нормативах»,
        «нормативных», «нормативы» только «нормативных» согласуется с
        вершиной «актов». Это снимает неоднозначность без языковой
        модели и без частотного словаря.
        """
        tokens = list(TOKEN_RE.finditer(text))
        index = next((i for i, m in enumerate(tokens) if m.start() == match.start()), None)
        if index is None:
            return False

        def gap_ok(left: re.Match[str], right: re.Match[str]) -> bool:
            return not any(ch in CLAUSE_BREAK for ch in text[left.end():right.start()])

        # Вариант как определение при ближайшем существительном справа.
        if self.morph.attributive_parses(variant):
            probe = index + 1
            while probe < len(tokens) and gap_ok(tokens[probe - 1], tokens[probe]):
                candidate = tokens[probe].group(0)
                if self.morph.noun_parses(candidate) and not self.morph.has_function_reading(candidate):
                    if self.morph.pair_agrees(variant, candidate):
                        return True
                    break
                if not self.morph.attributive_parses(candidate):
                    break
                probe += 1

        # Вариант как вершина при ближайшем определении слева.
        if self.morph.noun_parses(variant) and index > 0 and gap_ok(tokens[index - 1], tokens[index]):
            left = tokens[index - 1].group(0)
            if self.morph.attributive_parses(left) and self.morph.pair_agrees(left, variant):
                return True
        return False

    def _single_dictionary_fix(self, text: str, match: re.Match[str]) -> str | None:
        lowered = match.group(0).lower()
        variants = sorted(v for v in self._edits1(lowered) if len(v) >= 2 and self._known(v))
        # «ё»-варианты того же слова опечаткой не считаются.
        variants = [v for v in variants if v.replace("ё", "е") != lowered.replace("ё", "е")]
        variants = [v for v in variants if self._is_standard_form(v)]
        if len(variants) == 1:
            return variants[0]
        if not variants:
            return None
        in_context = [v for v in variants if self._agrees_with_context(text, match, v)]
        return in_context[0] if len(in_context) == 1 else None

    def _skip(self, text: str, match: re.Match[str]) -> bool:
        word = match.group(0)
        if len(word) < self.min_length or len(word) > self.max_length:
            return True
        if word.casefold() in self.protected:
            return True
        if word.isupper():  # аббревиатуры: ГУВД, ФСВНГ
            return True
        if "-" in word:
            return True
        if word[:1].isupper() and not SENTENCE_START_RE.search(text[:match.start()]):
            return True  # имена собственные внутри предложения
        return False

    def candidates(self, text: str) -> list[EditCandidate]:
        if not self.available or not text:
            return []
        out: list[EditCandidate] = []
        for match in TOKEN_RE.finditer(text):
            word = match.group(0)
            if self._skip(text, match):
                continue
            if self._known(word.lower()):
                continue
            fixed = self._single_dictionary_fix(text, match)
            if not fixed:
                continue
            if word[:1].isupper():
                fixed = fixed[:1].upper() + fixed[1:]
            if fixed == word:
                continue
            out.append(EditCandidate(
                word, fixed, self.confidence, "dict-spell",
                "слова нет в морфологическом словаре, словарный вариант единственный",
                start=match.start(),
            ))
        return out
