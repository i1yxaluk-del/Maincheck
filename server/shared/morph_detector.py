"""Морфологический ДЕТЕКТОР грамматических ошибок (v1.8a).

В отличие от `morph_filter.py` (который УДАЛЯЕТ галлюцинированные правки
модели), этот модуль **ДЕТЕКТИРУЕТ** реальные ошибки в исходном тексте,
которые модель T-lite пропустила, и предлагает их добавить в CHANGES.

Закрывает три класса prod-ошибок, обнаруженных на extension-тестах
(апрель–май 2026):

1. **Numeral-noun disagreement**: «во 2-м кварталах» — порядковое
   числительное в sing.loct + сущ. в plur.loct → ошибка согласования.
   T-lite этого не ловит даже с few-shot Lexify-case_agreement.

2. **Adj/Participle-noun disagreement**: «капитальных ремонтова» —
   adj.plur + noun.sing → disagreement. (Совпадает с детектором роли,
   что и в v1.7.3 для filter'а, но используется в обратном направлении.)

3. **OOV (out-of-vocabulary)**: слово, которое модель «выдумала» как
   правку или которое было ошибочно введено пользователем; pymorphy3
   не находит его в словаре (`is_known=False` для всех парсов). Кроме
   собственных имён (фамилий, географических названий) — их в pymorphy3
   часто нет в словаре, но это не ошибка.

Работа детектора:
  detector = get_morph_detector()
  errors = detector.detect_errors(raw_text)
  # errors: list[GrammarError] с полями offset, length, before, suggestion, kind

Если pymorphy3 не загружен — детектор no-op, возвращает [].

Интеграция: после T-lite в `_drop_*` пайплайне — `_enrich_changes_with_detector`
обогащает CHANGES пунктами, не пересекающимися с теми, что уже отдала модель.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

_log = logging.getLogger("ai_suggester.morph_detector")


# Граммемы, при наличии которых хотя бы у одного парса слово НЕ
# флагуется как OOV — это собственные имена, географические названия,
# организации, торговые марки. Аббревиатуры обрабатываем отдельной
# эвристикой по регистру (Abbr-тег pymorphy3 клеит и к мусорным
# словам типа «бярюфьл», поэтому только Abbr-тегу доверять нельзя).
_OOV_SAFE_TAG_MARKERS = frozenset({
    "Surn",   # фамилии (Иванов, Петров и т.д.)
    "Name",   # имена
    "Patr",   # отчества
    "Geox",   # географические названия
    "Orgn",   # организации
    "Trad",   # торговые марки
    "Init",   # инициалы
})


def _looks_like_abbreviation(word: str) -> bool:
    """Эвристика: ВЕРХНИЙ РЕГИСТР, 2+ букв, или с дефисом и цифрой.

    Закрывает: ЦСН, УФ, КС-2, МЧС, ФСБ, СВУ, FBI и т.д. — частые
    аббревиатуры в админ-документах. Слова с одной заглавной (Иванов)
    через эту эвристику НЕ проходят (там сработает Surn-парс).
    """
    if not word:
        return False
    # Содержит цифру и дефис — типа КС-2, ГОСТ-12345 и т.д.
    if any(c.isdigit() for c in word) and "-" in word:
        return True
    # Все буквы заглавные (минимум 2 буквы)
    letters = [c for c in word if c.isalpha()]
    if len(letters) >= 2 and all(c.isupper() for c in letters):
        return True
    return False

# Метаданные для совместимости с _CHANGE_PAIR_RE серверного парсинга.
# Кавычки — те же что использует модель в CHANGES.
_QUOTE_LEFT = "«"
_QUOTE_RIGHT = "»"


@dataclass(frozen=True)
class GrammarError:
    """Найденная морфо-детектором ошибка."""
    offset: int           # byte-offset в raw_text
    length: int           # длина проблемного фрагмента (в символах)
    before: str           # как написано (фрагмент)
    suggestion: str       # что предложить (одна форма)
    kind: str             # тип: "numeral_noun" | "adj_noun" | "oov"
    explanation: str      # короткое объяснение для CHANGES

    def to_change_line(self, number: int) -> str:
        """Сериализация в формат, совпадающий с _CHANGE_PAIR_RE."""
        return (
            f"{number}. {_QUOTE_LEFT}{self.before}{_QUOTE_RIGHT} → "
            f"{_QUOTE_LEFT}{self.suggestion}{_QUOTE_RIGHT} | {self.explanation}"
        )


# Word tokenizer with offsets — нужен offset чтобы вернуть позицию в
# raw_text. Регэксп ловит русские слова (включая «2-м», «5-х» — слова
# с цифрами и дефисом, обычные для деловых текстов).
_WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+(?:[-/][А-Яа-яЁёA-Za-z0-9]+)*")


def _iter_words_with_offsets(text: str):
    """Итератор (offset, word) для всех словесных токенов в `text`.

    Группа отделена от обычных whitespace и пунктуации. «2-м», «КС-2»
    и т.п. остаются единой токеной — это нужно для морф-разбора.
    """
    for m in _WORD_RE.finditer(text):
        yield m.start(), m.group(0)


class MorphDetector:
    """Детектор морфологических ошибок через pymorphy3.

    API:
      * `detect_errors(raw_text) -> list[GrammarError]` — главный метод.
      * `available` — True если pymorphy3 загружен; иначе детектор no-op.

    Singleton pattern: используйте `get_morph_detector()` для повторного
    использования инстанса (pymorphy3 жадный по памяти при инициализации).
    """

    def __init__(self) -> None:
        self._morph = None
        try:
            import pymorphy3  # type: ignore[import-untyped]
            self._morph = pymorphy3.MorphAnalyzer()
            _log.info("MorphDetector: pymorphy3 загружен (готов детектировать ошибки)")
        except Exception as e:
            _log.warning(
                "MorphDetector: pymorphy3 не загружен (%s) — детектор отключён",
                e,
            )

    @property
    def available(self) -> bool:
        return self._morph is not None

    # ───────────────────────────────────────────────────────────────────
    # Numeral-noun agreement detector
    # ───────────────────────────────────────────────────────────────────

    def _has_compatible_parse_pair(
        self, word_a: str, word_b: str,
        check_number: bool = True,
        check_case: bool = True,
        check_gender: bool = False,
    ) -> bool:
        """True если есть хотя бы одна пара (parse_a, parse_b) где
        указанные граммемы совпадают.

        Используется для проверки «согласовано ли A с B?». Если ни одна
        пара парсов не даёт совместимости — слова рассогласованы.
        """
        if not self._morph:
            return True  # консервативно — не флагуем, фильтр выкл.
        parses_a = self._morph.parse(word_a)
        parses_b = self._morph.parse(word_b)
        for pa in parses_a:
            for pb in parses_b:
                ok = True
                if check_number and pa.tag.number != pb.tag.number:
                    ok = False
                if ok and check_case and pa.tag.case != pb.tag.case:
                    ok = False
                if ok and check_gender:
                    # gender проверяем только если оба известны
                    if pa.tag.gender and pb.tag.gender:
                        if pa.tag.gender != pb.tag.gender:
                            ok = False
                if ok:
                    return True
        return False

    def _is_ordinal_numeral(self, word: str) -> bool:
        """True если хотя бы один парс — порядковое числительное.

        В pymorphy3 это ADJF с маркерами Anum (числительное-прилагательное)
        либо чисто `NUMR` (количественное). Мы интересуемся только
        ADJF.Anum или ADJF + Numr-style потому что только они идут в
        паттерне «во 2-м квартале / в 5-х случаях».
        """
        if not self._morph:
            return False
        parses = self._morph.parse(word)
        for p in parses:
            tag_str = str(p.tag)
            if "Anum" in tag_str or "NUMR" in tag_str:
                return True
        return False

    def _is_noun(self, word: str) -> bool:
        """True если хотя бы один парс — существительное."""
        if not self._morph:
            return False
        parses = self._morph.parse(word)
        return any("NOUN" in str(p.tag) for p in parses)

    def _is_known_word(self, word: str) -> bool:
        """True если слово есть в словаре pymorphy3.

        v1.8.1: используется в детекторах adj_noun/numeral_noun, чтобы не
        срабатывать на парах с OOV-словом (например «ремонтова» парсится
        как ADJS/Surn через FakeDictionary, но это ложная интерпретация).
        OOV-детектор пометит такое слово отдельно как «oov».
        """
        if not self._morph:
            return True  # нет анализатора — не блокируем остальные проверки
        try:
            return self._morph.word_is_known(word)
        except Exception:
            return True

    def _suggest_noun_form(
        self, noun_word: str, target_number: Optional[str], target_case: Optional[str],
    ) -> Optional[str]:
        """Подбирает форму существительного `noun_word` с заданными
        числом и падежом. None если pymorphy не смог.
        """
        if not self._morph or (target_number is None and target_case is None):
            return None
        parses = self._morph.parse(noun_word)
        if not parses:
            return None
        # Берём best-парс и просим его inflect()
        best = parses[0]
        grammemes = set()
        if target_number:
            grammemes.add(target_number)
        if target_case:
            grammemes.add(target_case)
        try:
            inflected = best.inflect(grammemes)
        except Exception:
            return None
        if not inflected:
            return None
        # Сохраняем регистр первой буквы исходного слова
        result = inflected.word
        if noun_word and noun_word[0].isupper():
            result = result[0].upper() + result[1:]
        return result

    def detect_numeral_noun_disagreements(
        self, raw_text: str,
    ) -> list[GrammarError]:
        """Находит пары (порядковое-числительное, существительное), где
        число различается. Главный prod-кейс: «во 2-м кварталах».

        Стратегия:
          1. Идём по словам. Для каждой пары (i, i+1):
          2. Если word_i — ordinal numeral, word_{i+1} — noun, и нет
             совместимого парса по number+case → disagreement.
          3. Подбираем suggestion: меняем число существительного на то,
             что у numeral'а (по best-parse).
        """
        if not self.available:
            return []
        errors: list[GrammarError] = []
        words = list(_iter_words_with_offsets(raw_text))
        for i in range(len(words) - 1):
            offset_a, word_a = words[i]
            offset_b, word_b = words[i + 1]
            if not self._is_ordinal_numeral(word_a):
                continue
            if not self._is_noun(word_b):
                continue
            # v1.8.1: пропускаем если любое из слов — OOV (форма из FakeDictionary).
            # Такие кейсы обрабатываются detect_oov_words, не adj/numeral.
            if not self._is_known_word(word_a) or not self._is_known_word(word_b):
                continue
            # Проверяем согласование по number+case
            if self._has_compatible_parse_pair(
                word_a, word_b, check_number=True, check_case=True,
            ):
                continue
            # Disagreement — подбираем форму noun с number+case от numeral
            num_parse = self._morph.parse(word_a)[0]  # type: ignore[union-attr]
            target_number = num_parse.tag.number
            target_case = num_parse.tag.case
            suggestion = self._suggest_noun_form(word_b, target_number, target_case)
            if not suggestion or suggestion == word_b:
                continue
            errors.append(GrammarError(
                offset=offset_b,
                length=len(word_b),
                before=word_b,
                suggestion=suggestion,
                kind="numeral_noun",
                explanation=(
                    "согласование существительного с числительным по числу"
                ),
            ))
        return errors

    # ───────────────────────────────────────────────────────────────────
    # Adjective/Participle-noun agreement detector
    # ───────────────────────────────────────────────────────────────────

    def _is_adj_or_participle(self, word: str) -> bool:
        """True если хотя бы один парс — ADJF/ADJS/PRTF/PRTS."""
        if not self._morph:
            return False
        parses = self._morph.parse(word)
        for p in parses:
            tag_str = str(p.tag)
            if any(t in tag_str for t in ("ADJF", "ADJS", "PRTF", "PRTS")):
                return True
        return False

    def _is_participle(self, word: str) -> bool:
        """True если хотя бы один парс — PRTF/PRTS (только причастие).

        Отличается от `_is_adj_or_participle` тем, что чистые ADJF/ADJS
        НЕ считаются причастиями. Используется для целевого исключения
        паттерна «причастие + агенс_в_творительном» в детекторе adj_noun
        (см. v1.8.5).
        """
        if not self._morph:
            return False
        parses = self._morph.parse(word)
        for p in parses:
            tag_str = str(p.tag)
            if "PRTF" in tag_str or "PRTS" in tag_str:
                return True
        return False

    def _can_be_instrumental(self, word: str) -> bool:
        """True если у слова есть хотя бы один парс в творительном падеже
        (`ablt`).

        Используется в `detect_adj_noun_disagreements` для подавления
        FP-класса «причастие + агенс_в_творительном» (v1.8.5).
        """
        if not self._morph:
            return False
        parses = self._morph.parse(word)
        for p in parses:
            if p.tag.case == "ablt":
                return True
        return False

    def _is_preposition(self, word: str) -> bool:
        """True если хотя бы один парс слова — предлог (PREP).

        Используется в `detect_adj_noun_disagreements` для подавления
        FP-класса «при этом X» / «над тем Y»: если adj/прич перед NOUN
        управляется предшествующим предлогом, оно НЕ модифицирует NOUN,
        а образует с предлогом prepositional-фразу (v1.8.5).
        """
        if not self._morph:
            return False
        # Перепроверяем lower-case — предлоги пишутся всегда строчно
        # кроме начала предложения. Pymorphy3 распознаёт «при» и «При»
        # одинаково, но lower-case надёжнее.
        parses = self._morph.parse(word.lower())
        for p in parses:
            if "PREP" in str(p.tag):
                return True
        return False

    def _is_short_form_only(self, word: str) -> bool:
        """True если у слова ВСЕ парсы — краткая форма (ADJS/PRTS).
        Краткие формы не склоняются по падежам, поэтому в пред-позиции
        («указаны работы») case-согласование не применимо.
        """
        if not self._morph:
            return False
        parses = self._morph.parse(word)
        if not parses:
            return False
        for p in parses:
            tag_str = str(p.tag)
            if "ADJF" in tag_str or "PRTF" in tag_str:
                return False
            # ADVB / NOUN / VERB → значит у слова есть варианты вне ADJS/PRTS
            if not ("ADJS" in tag_str or "PRTS" in tag_str):
                return False
        return True

    def detect_adj_noun_disagreements(
        self, raw_text: str,
    ) -> list[GrammarError]:
        """Находит пары (adj/participle, noun), где число различается
        (или падеж рассогласован).

        Главные кейсы:
          * «капитальных ремонтова» — adj.plur.gent + (если бы корректное
             слово) noun.sing.gent → disagreement.
          * «Проверочное мероприятия» — adj.sing.neut.nomn + noun.gent.sing
             либо noun.plur — disagreement.

        Чтобы не дублировать логику morph_filter v1.7.3 (которая работает
        для GRAMMAR-направления — фильтрует false-positive галлюцинации),
        эта функция работает в DETECTION-направлении: ищет ошибки в
        исходном raw_text, которых модель не пофиксила.
        """
        if not self.available:
            return []
        errors: list[GrammarError] = []
        words = list(_iter_words_with_offsets(raw_text))
        for i in range(len(words) - 1):
            offset_a, word_a = words[i]
            offset_b, word_b = words[i + 1]
            if not self._is_adj_or_participle(word_a):
                continue
            if not self._is_noun(word_b):
                continue
            # ВАЖНО: пропускаем если adj можно интерпретировать как
            # числительное (это уже ловит numeral_noun детектор).
            if self._is_ordinal_numeral(word_a):
                continue
            # v1.8.1: пропускаем если любое из слов — OOV. Был prod-FP на
            # «капитальных ремонтова помещений»: «ремонтова» (OOV) парсился
            # как ADJS через FakeDictionary, и детектор выдавал два FP по этой
            # ложной интерпретации. OOV-детектор отловит «ремонтова» отдельно.
            if not self._is_known_word(word_a) or not self._is_known_word(word_b):
                continue
            # v1.8.5: пропускаем паттерн «причастие + сущ_в_творительном».
            # В русском причастие управляет агенсом в творительном падеже:
            # «проводимых подразделениями» (carried out by subdivisions),
            # «открытое отделом» (opened by department), «решённый комиссией»
            # (resolved by commission). В таких парах причастие НЕ
            # согласуется с этим существительным — оно согласуется с
            # upstream-головой (например, «мероприятий, проводимых X»).
            # Без этой проверки детектор флагует подавляющую часть таких
            # пар как disagreement — критический FP-класс в admin-текстах
            # (прод-кейс 05.05.2026: «проводимых подразделениями» →
            # «проводимых подразделений», что грамматически неверно).
            # Ограничение: ADJF/ADJS тут НЕ исключаются — они редко
            # управляют творительным как агенсом («довольный отделом»
            # остаётся как минор-FP, разбираться отдельно).
            if self._is_participle(word_a) and self._can_be_instrumental(word_b):
                continue
            # v1.8.5: пропускаем паттерн «<предлог> <местоим/прил> <сущ>»,
            # где «местоим/прил» — это объект предлога, а не модификатор
            # следующего сущ. Прод-кейс 05.05.2026:
            #   «...преступления – 99, при этом количество должностных...»
            # Пара (этом, количество): «этом» (loct.neut.sing) флагуется
            # как disagreement с «количество» (nomn.sing.neut). Но «при
            # этом» — discourse marker («moreover»), «этом» бнут к «при»
            # (PREP loct), а не к «количество». Аналогичные паттерны:
            #   «о том X», «о тех Y», «в этом N», «над этой M».
            # Проверяем что предыдущее слово (i-1) — это предлог.
            # Ограничение: legitimate-FP остаётся на ситуациях вроде
            # «над новой ошибкой» (где «новая» правильно agrees с
            # «ошибкой») — но они НЕ дают disagreement в _has_compatible,
            # так что эта проверка их не блокирует.
            if i > 0:
                _, prev_word_outer = words[i - 1]
                if self._is_preposition(prev_word_outer):
                    continue
            # Краткие формы (ADJS/PRTS) — predicative, не склоняются
            # по падежам. Для них проверяем только number+gender.
            if self._is_short_form_only(word_a):
                if self._has_compatible_parse_pair(
                    word_a, word_b,
                    check_number=True, check_case=False, check_gender=True,
                ):
                    continue
            else:
                # Атрибутивная позиция: проверяем number+case+gender
                if self._has_compatible_parse_pair(
                    word_a, word_b,
                    check_number=True, check_case=True, check_gender=True,
                ):
                    continue
            # Disagreement найден.
            adj_parse = self._morph.parse(word_a)[0]  # type: ignore[union-attr]
            target_number = adj_parse.tag.number
            target_case = adj_parse.tag.case
            suggestion = self._suggest_noun_form(word_b, target_number, target_case)
            if not suggestion or suggestion == word_b:
                continue
            errors.append(GrammarError(
                offset=offset_b,
                length=len(word_b),
                before=word_b,
                suggestion=suggestion,
                kind="adj_noun",
                explanation=(
                    "согласование существительного с прилагательным по числу/роду"
                ),
            ))
        return errors

    # ───────────────────────────────────────────────────────────────────
    # OOV (out-of-vocabulary) detector
    # ───────────────────────────────────────────────────────────────────

    def detect_oov_words(
        self, raw_text: str, whitelist: Optional[frozenset[str]] = None,
    ) -> list[GrammarError]:
        """Находит слова, которых нет в словаре pymorphy3 (выдуманные
        формы вроде «ремонтова», «бярюфьл»). Не флагуем фамилии,
        собственные имена, географические названия и аббревиатуры.

        whitelist — пользовательский словарь (lower-case set);
        слова в нём НЕ флагуются как OOV.
        """
        if not self.available:
            return []
        errors: list[GrammarError] = []
        whitelist_lower = {w.lower() for w in (whitelist or frozenset())}
        for offset, word in _iter_words_with_offsets(raw_text):
            # Пропускаем чисто числовые токены, слова из 1-2 букв
            if word.isdigit() or len(word) <= 2:
                continue
            # Пропускаем если в пользовательском словаре
            if word.lower() in whitelist_lower:
                continue
            # Пропускаем если выглядит как аббревиатура (ЦСН, КС-2, МЧС)
            if _looks_like_abbreviation(word):
                continue
            parses = self._morph.parse(word)  # type: ignore[union-attr]
            if not parses:
                continue
            # Все парсы unknown? (is_known=False для каждого)
            all_unknown = all(not p.is_known for p in parses)
            if not all_unknown:
                continue
            # Если хотя бы один парс — собственное имя (фамилия/имя/гео),
            # не флагуем — НО только если этот парс из реального словаря
            # (is_known=True). v1.8.1: «ремонтова» получает Surn-парс через
            # FakeDictionary с is_known=False — этому доверять нельзя, иначе
            # любое выдуманное слово на «-ова», «-ин», «-ский» останется
            # неотловленным. Реальные фамилии (Иванов, Сорокин) парсятся как
            # Surn с is_known=True через DictionaryAnalyzer и здесь проходят.
            has_safe_tag = any(
                p.is_known
                and any(marker in str(p.tag) for marker in _OOV_SAFE_TAG_MARKERS)
                for p in parses
            )
            if has_safe_tag:
                continue
            # OOV — флаг
            errors.append(GrammarError(
                offset=offset,
                length=len(word),
                before=word,
                # Suggestion для OOV — пустая строка (модель/юзер должны
                # сами решить замену). В CHANGES представим как:
                # «слово отсутствует в словаре, проверьте написание».
                suggestion="",
                kind="oov",
                explanation=(
                    "слово отсутствует в словаре, проверьте написание"
                ),
            ))
        return errors

    # ───────────────────────────────────────────────────────────────────
    # Главный публичный метод
    # ───────────────────────────────────────────────────────────────────

    def detect_errors(
        self, raw_text: str, whitelist: Optional[frozenset[str]] = None,
    ) -> list[GrammarError]:
        """Запускает все детекторы и возвращает консолидированный
        список ошибок, отсортированный по offset.

        Дедупликация: если две ошибки указывают на тот же фрагмент
        (same offset+length) — оставляем первую (приоритет: numeral_noun
        > adj_noun > oov).
        """
        if not self.available:
            return []
        all_errors: list[GrammarError] = []
        all_errors.extend(self.detect_numeral_noun_disagreements(raw_text))
        all_errors.extend(self.detect_adj_noun_disagreements(raw_text))
        all_errors.extend(self.detect_oov_words(raw_text, whitelist))
        # Sort by offset, dedupe by (offset, length) — keep first.
        all_errors.sort(key=lambda e: (e.offset, e.length))
        deduped: list[GrammarError] = []
        seen: set[tuple[int, int]] = set()
        for err in all_errors:
            key = (err.offset, err.length)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(err)
        return deduped


# ─── Singleton ────────────────────────────────────────────────────────


_INSTANCE: Optional[MorphDetector] = None


def get_morph_detector() -> MorphDetector:
    """Возвращает singleton MorphDetector. pymorphy3 жадный по памяти
    при инициализации (~50 МБ словаря) — повторно используем инстанс.
    """
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = MorphDetector()
    return _INSTANCE
