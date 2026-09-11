"""Сегментация русского текста на предложения с сохранением смещений (v9).

Зачем
=====
v8 отправлял в SAGE и в GEC-специалиста весь выделенный фрагмент целиком.
Это плохо по трём причинам:

1. `sage-fredt5-distilled-95m` — seq2seq на 95 М параметров, обученный
   преимущественно на одиночных предложениях. На абзаце он начинает
   перефразировать, и `safe_diff._global_rewrite` выбрасывает результат
   целиком — стадия работает, тратит секунды CPU и не даёт правок.
2. Внимание квадратично по длине: два предложения по 100 символов
   считаются существенно быстрее, чем одно на 200, и считаются
   параллельно.
3. Ошибка в одном предложении обесценивала правки во всех остальных:
   один плохой diff → `_global_rewrite` → ноль кандидатов.

Сегментация учитывает особенности служебных документов: сокращения
(«п.», «ст.», «г.», «т.д.», «руб.»), инициалы («Иванов И.И.»), нумерацию
пунктов («1.», «1.2.»), перечисления через «;» и переносы строк.
Смещения сохраняются, поэтому правка из сегмента адресуется абсолютной
позицией в исходном тексте.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Сокращения, после точки которых предложение не заканчивается.
ABBREVIATIONS = frozenset({
    "п", "пп", "ст", "стст", "ч", "гл", "разд", "абз", "прим", "пример",
    "г", "гг", "в", "вв", "т", "тт", "тыс", "млн", "млрд", "руб", "коп",
    "им", "ул", "пр", "д", "корп", "стр", "обл", "респ", "р-н", "пос",
    "др", "пр-во", "зам", "и.о", "мин", "сек", "ч.", "шт", "экз",
    "напр", "см", "ср", "рис", "табл", "прил", "изд", "ред", "сост",
    "проф", "доц", "канд", "докт", "акад", "тел", "факс", "эл",
})

_SENTENCE_END = re.compile(r"([.!?;…]+|\n{1,})(\s+|$)")
_WORD_BEFORE_DOT = re.compile(r"([А-Яа-яЁёA-Za-z]+)\.$")
_INITIAL = re.compile(r"\b[А-ЯЁ]\.$")
_ENUMERATION = re.compile(r"^\s*(?:\d+[.)]|[а-яё][.)]|[-—•])\s*")
_ONLY_NUMBERING = re.compile(r"\s*(?:\d+[.)])*\d+\.\s*$")
_LOWERCASE_START = re.compile(r"^[а-яёa-z]")


@dataclass(frozen=True)
class Segment:
    """Фрагмент текста и его абсолютное смещение в исходной строке."""

    text: str
    start: int

    @property
    def end(self) -> int:
        return self.start + len(self.text)


def _is_hard_break(text: str, match: re.Match[str], cursor: int) -> bool:
    terminator = match.group(1)
    if "\n" in terminator:
        return True
    prefix = text[:match.start(1) + 1]
    if _INITIAL.search(prefix):
        return False  # «Иванов И.И. Замечаний нет.»
    word = _WORD_BEFORE_DOT.search(prefix)
    if word and word.group(1).casefold() in ABBREVIATIONS:
        return False  # «п. 3.2», «ст. 15», «т. д.»
    if terminator == "." and _ONLY_NUMBERING.fullmatch(text[cursor:match.end(1)]):
        return False  # маркер пункта в начале строки: «1.», «1.2.»
    if terminator == "." and _LOWERCASE_START.match(text[match.end():]):
        return False  # «...от 01.02.2026 г. составлен акт»
    return True


def split_sentences(text: str, min_length: int = 1) -> list[Segment]:
    """Разбивает текст на предложения, сохраняя абсолютные смещения.

    Границы сохраняют исходную пунктуацию и пробелы внутри сегмента, так
    что конкатенация сегментов с исходными промежутками восстанавливает
    текст без потерь.
    """
    if not text:
        return []
    segments: list[Segment] = []
    cursor = 0
    for match in _SENTENCE_END.finditer(text):
        if not _is_hard_break(text, match, cursor):
            continue
        end = match.end(1)
        raw = text[cursor:end]
        stripped = raw.strip()
        if len(stripped) >= min_length:
            offset = cursor + (len(raw) - len(raw.lstrip()))
            segments.append(Segment(raw.strip(), offset))
        cursor = match.end()
    tail = text[cursor:]
    if tail.strip():
        offset = cursor + (len(tail) - len(tail.lstrip()))
        segments.append(Segment(tail.strip(), offset))
    return segments or [Segment(text.strip(), len(text) - len(text.lstrip()))]


def strip_enumeration(segment: Segment) -> Segment:
    """Отделяет маркер перечисления: «1. Проверка проведена» → «Проверка…».

    Модели устойчивее работают без номера пункта, а номер при этом
    гарантированно не может быть изменён.
    """
    match = _ENUMERATION.match(segment.text)
    if not match or match.end() >= len(segment.text):
        return segment
    return Segment(segment.text[match.end():], segment.start + match.end())
