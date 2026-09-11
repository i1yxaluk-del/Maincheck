"""Морфологическое ядро v9: корректная проверка русского согласования.

Зачем отдельный модуль
======================
До v9 согласование проверялось внутри `local_rules._modifier_noun` по
схеме «взять `parse[0]` у существительного, взять `parse[0]` у
прилагательного, вызвать `inflect({case, number, gender})`». На
официально-деловом русском это давало ложные срабатывания практически
на каждом втором предложении, потому что:

1. **Омонимия падежей не разрешается выбором `parse[0]`.** У «раздел»
   pymorphy3 даёт `accs` (score 0.50) и `nomn` (score 0.43). Выбор
   первого парса объявляет винительный падеж единственно верным, и
   согласованное «соответствующий раздел» признаётся ошибкой.

2. **`inflect` без грамммемы одушевлённости ломает винительный падеж
   мужского рода.** `inflect({"accs", "sing", "masc"})` возвращает
   одушевлённую форму, совпадающую с родительным: «соответствующий» →
   «соответствующего». Это и есть дефект из прод-отчёта.

3. **Не проверялось, согласована ли пара уже сейчас.** Код перебирал
   парсы до первого, дающего *отличающуюся* форму, т.е. был системно
   смещён в сторону порождения правки.

4. **Краткие формы (ADJS/PRTS) считались атрибутивными.** «Утвержден
   план» → «Утверждённого план»: краткое причастие — это сказуемое, оно
   никогда не согласуется по падежу.

5. **`ё` дописывалась к словам, где источник использует `е`.»
   «отчетный» → «отчётного»: помимо падежа менялась орфография ё/е,
   чего пользователь не просил.

Принцип v9
==========
Правка предлагается только если **ни одна** пара парсов (модификатор,
вершина) не согласуется. Если существует хотя бы одно прочтение, при
котором текст корректен, — текст корректен. Это стандартный для
морфологической омонимии подход «any-reading-valid» и он переводит
компонент из режима «ловим всё» в режим «высокая точность».
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Sequence

log = logging.getLogger("ai_suggester.morphology")

# pymorphy3 отдаёт вторичные падежи отдельными грамммемами. Для проверки
# согласования они эквивалентны основным.
CASE_ALIASES = {
    "gen1": "gent",
    "gen2": "gent",
    "acc2": "accs",
    "loc1": "loct",
    "loc2": "loct",
}

#: Полные формы, способные быть атрибутивным определением.
ATTRIBUTIVE_POS = frozenset({"ADJF", "PRTF"})

#: Краткие формы — предикативы, по падежу не согласуются.
PREDICATIVE_POS = frozenset({"ADJS", "PRTS"})

#: POS, из-за которых слово нельзя считать вершиной именной группы.
FUNCTION_POS = frozenset({"PREP", "CONJ", "PRCL", "INTJ"})

NOUN_POS = frozenset({"NOUN"})


@dataclass(frozen=True)
class Features:
    """Нормализованные признаки словоформы."""

    pos: str | None
    case: str | None
    number: str | None
    gender: str | None
    animacy: str | None

    def agrees_with(self, head: "Features") -> bool:
        """True, если эта форма может быть согласованным определением `head`."""
        if self.number and head.number and self.number != head.number:
            return False
        if self.case and head.case and self.case != head.case:
            return False
        # В множественном числе род у прилагательных не выражается.
        plural = "plur" in {self.number, head.number}
        if not plural and self.gender and head.gender and self.gender != head.gender:
            return False
        # В винительном падеже прилагательное различает одушевлённость, и
        # одушевлённость вершины — лексический факт. Без этой проверки
        # «соответствующего раздел» считалось согласованным: у формы
        # «соответствующего» есть разбор accs+anim, а у «раздел» —
        # accs+inan.
        if (
            self.case == "accs"
            and self.animacy
            and head.animacy
            and self.animacy != head.animacy
        ):
            return False
        return True


def _alias(case: str | None) -> str | None:
    return CASE_ALIASES.get(case, case) if case else None


def features(parse: Any) -> Features:
    tag = parse.tag
    return Features(
        pos=tag.POS,
        case=_alias(tag.case),
        number=tag.number,
        gender=tag.gender,
        animacy=tag.animacy,
    )


class Morphology:
    """Тонкая обёртка над pymorphy3 с кэшированием разборов.

    `available == False`, если pymorphy3 не установлен; в этом случае все
    методы возвращают пустые результаты, а вызывающий код деградирует до
    работы без морфологии, вместо падения.
    """

    def __init__(self) -> None:
        self._morph = None
        try:
            import pymorphy3

            self._morph = pymorphy3.MorphAnalyzer()
        except Exception as exc:  # pragma: no cover - окружение без словарей
            log.warning("Morphology: pymorphy3 недоступен (%s)", exc)

    @property
    def available(self) -> bool:
        return self._morph is not None

    @lru_cache(maxsize=8192)
    def _parse_cached(self, word: str) -> tuple:
        if self._morph is None:
            return ()
        try:
            return tuple(self._morph.parse(word))
        except Exception:  # pragma: no cover
            return ()

    def parse(self, word: str) -> tuple:
        if not word:
            return ()
        return self._parse_cached(word)

    def known_parses(self, word: str) -> list[Any]:
        return [p for p in self.parse(word) if p.is_known]

    def is_known(self, word: str) -> bool:
        return bool(self.known_parses(word))

    def parses_with_pos(self, word: str, allowed: Iterable[str]) -> list[Any]:
        allowed = set(allowed)
        return [p for p in self.known_parses(word) if p.tag.POS in allowed]

    def attributive_parses(self, word: str) -> list[Any]:
        """Полные прилагательные/причастия/адъективные местоимения."""
        return self.parses_with_pos(word, ATTRIBUTIVE_POS)

    def noun_parses(self, word: str) -> list[Any]:
        return self.parses_with_pos(word, NOUN_POS)

    def has_function_reading(self, word: str) -> bool:
        """True, если слово может быть предлогом/союзом/частицей.

        Такие токены нельзя трактовать как вершину именной группы: у
        pymorphy3 «в», «а», «то», «и» имеют и NOUN-разборы (названия
        букв), из-за чего старый код строил согласование «устранены» с
        «существительным "в"».
        """
        return any(p.tag.POS in FUNCTION_POS for p in self.parse(word))

    def is_predicative(self, word: str) -> bool:
        parses = self.known_parses(word)
        if not parses:
            return False
        return all(p.tag.POS in PREDICATIVE_POS for p in parses if p.tag.POS)

    def lemmas(self, word: str) -> set[str]:
        return {p.normal_form for p in self.known_parses(word)}

    # ------------------------------------------------------------------
    # Согласование
    # ------------------------------------------------------------------
    def pair_agrees(self, modifier: str, head: str) -> bool:
        """True, если существует прочтение, при котором пара согласована.

        Это главный предохранитель от ложных правок: любая
        морфологическая омонимия трактуется в пользу исходного текста.
        """
        mods = self.attributive_parses(modifier)
        heads = self.noun_parses(head)
        if not mods or not heads:
            return True  # нет данных — считаем текст корректным
        for m in mods:
            mf = features(m)
            for h in heads:
                if mf.agrees_with(features(h)):
                    return True
        return False

    def agreeing_head_parses(self, modifier: str, head_parses: Sequence[Any]) -> list[Any]:
        """Разборы вершины, с которыми модификатор согласован."""
        mods = [features(m) for m in self.attributive_parses(modifier)]
        if not mods:
            return []
        return [h for h in head_parses if any(m.agrees_with(features(h)) for m in mods)]

    # ------------------------------------------------------------------
    # Словоизменение
    # ------------------------------------------------------------------
    @staticmethod
    def target_grammemes(head: Any) -> set[str]:
        """Грамммемы для согласования определения с данной вершиной.

        Одушевлённость включается обязательно: без неё `inflect` для
        винительного падежа мужского рода отдаёт одушевлённую форму
        («соответствующего» вместо «соответствующий»).
        """
        f = features(head)
        target: set[str] = set()
        if f.case:
            target.add(f.case)
        if f.number:
            target.add(f.number)
        if f.number != "plur" and f.gender:
            target.add(f.gender)
        if f.animacy and (f.case == "accs" or f.number == "plur"):
            target.add(f.animacy)
        return target

    def inflect_modifier(self, modifier: str, head: Any) -> str | None:
        """Форма `modifier`, согласованная с разбором `head`.

        Возвращает None, если согласованная форма не найдена, совпадает
        с исходной или не проходит обратную проверку согласования.
        """
        target = self.target_grammemes(head)
        if not target:
            return None
        head_features = features(head)
        for parse in self.attributive_parses(modifier)[:8]:
            try:
                inflected = parse.inflect(target)
            except Exception:  # pragma: no cover
                inflected = None
            if not inflected or not inflected.word:
                continue
            produced = inflected.word
            if produced.lower() == modifier.lower():
                continue
            # Обратная проверка: результат обязан согласоваться с вершиной.
            if not features(inflected).agrees_with(head_features):
                continue
            produced = preserve_yo(modifier, produced)
            produced = preserve_capitalization(modifier, produced)
            if produced == modifier:
                continue
            return produced
        return None

    def inflected_forms(self, word: str, grammemes: set[str]) -> set[str]:
        out: set[str] = set()
        for parse in self.known_parses(word)[:8]:
            try:
                form = parse.inflect(grammemes)
            except Exception:  # pragma: no cover
                form = None
            if form and form.word:
                out.add(form.word)
        return out


def preserve_yo(source: str, produced: str) -> str:
    """Сохраняет ё/е-конвенцию исходного слова.

    Словарь pymorphy3 всегда отдаёт формы с «ё». Если в исходном слове
    «ё» не использовалась, дописывать её нельзя: это не исправление
    ошибки, а смена орфографической конвенции документа.
    """
    if "ё" in source.lower():
        return produced
    return produced.replace("ё", "е").replace("Ё", "Е")


def preserve_capitalization(source: str, produced: str) -> str:
    if not source or not produced:
        return produced
    if source.isupper() and len(source) > 1:
        return produced.upper()
    if source[:1].isupper():
        return produced[:1].upper() + produced[1:]
    return produced


def yo_equal(left: str, right: str) -> bool:
    return left.lower().replace("ё", "е") == right.lower().replace("ё", "е")


_SINGLETON: Morphology | None = None


def get_morphology() -> Morphology:
    """Общий экземпляр: словари pymorphy3 занимают ~50 МБ."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = Morphology()
    return _SINGLETON
