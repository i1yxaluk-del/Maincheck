"""Извлечение локальных правок из пары «исходный текст → вывод модели».

Почему по токенам, а не по символам
===================================
v8 сравнивал строки посимвольно (`SequenceMatcher` по `str`). На правке
«нескольких вида» → «нескольких видов» это давало кандидата `«а» → «ов»`
с двумя последствиями:

* фрагмент `«а»` встречается в тексте много раз, и `DecisionEngine`
  отбрасывал правку как неоднозначную — реальное исправление терялось;
* в блоке `===CHANGES===` пользователь видел `«а» → «ов»` вместо
  понятного `«вида» → «видов»`.

v9 сравнивает последовательности токенов (слова, числа, знаки) и
возвращает правки, выровненные по границам слов. Это одновременно
улучшает читаемость и позволяет морфологическому фильтру
(`verification.GenerativeGuard`) сопоставлять слова попарно.
"""

from __future__ import annotations

import difflib
import math
import re

from decision_engine import EditCandidate

WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z]+")
TOKEN_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+|[^\sА-Яа-яЁёA-Za-z0-9]")

MAX_FRAGMENT_CHARS = 90


def _tokens(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def _global_rewrite(source: str, corrected: str) -> bool:
    """Отклоняет пересказ и перестановку: допускаем лишь точечные правки."""
    source_words = WORD_RE.findall(source)
    corrected_words = WORD_RE.findall(corrected)
    if source_words and abs(len(corrected_words) - len(source_words)) > 1:
        return True
    baseline = max(1, min(len(source), len(corrected)))
    if abs(len(corrected) - len(source)) / baseline > 0.20:
        return True

    matcher = difflib.SequenceMatcher(None, source_words, corrected_words, autojunk=False)
    changed = 0
    replace_ops = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed += max(i2 - i1, j2 - j1)
        if tag == "replace":
            replace_ops += 1
        if (i2 - i1) > 2 or (j2 - j1) > 2:
            return True
    allowed = max(3, math.ceil(max(len(source_words), 1) * 0.12))
    return changed > allowed or replace_ops > 3


def diff_candidates(source: str, corrected: str, category: str, confidence: float = 0.70,
                    offset: int = 0) -> list[EditCandidate]:
    """Локальные правки с абсолютными смещениями.

    `offset` — смещение `source` в тексте запроса. Модели работают по
    предложениям, и без смещения правку пришлось бы искать подстрокой.
    """
    if not source or not corrected or source == corrected:
        return []
    if source.count("\n") != corrected.count("\n"):
        return []
    if len(source.split("\n\n")) != len(corrected.split("\n\n")):
        return []
    if _global_rewrite(source, corrected):
        return []

    src = _tokens(source)
    dst = _tokens(corrected)
    if not src or not dst:
        return []

    opcodes = difflib.SequenceMatcher(
        None, [t[0] for t in src], [t[0] for t in dst], autojunk=False,
    ).get_opcodes()

    out: list[EditCandidate] = []
    for position, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        if tag == "equal":
            continue
        # Вставка и удаление не имеют собственного якоря в тексте:
        # расширяем фрагмент на один соседний токен с каждой стороны,
        # чтобы правка однозначно привязывалась к позиции.
        if i1 == i2 or j1 == j2:
            i1 = max(0, i1 - 1)
            i2 = min(len(src), i2 + 1)
            j1 = max(0, j1 - 1)
            j2 = min(len(dst), j2 + 1)
        if i1 >= i2 or j1 >= j2:
            return []

        start, end = src[i1][1], src[i2 - 1][2]
        before = source[start:end]
        after = corrected[dst[j1][1]:dst[j2 - 1][2]]
        if "\n" in before or "\n" in after:
            return []
        if not before or not after:
            return []
        if len(before) > MAX_FRAGMENT_CHARS or len(after) > MAX_FRAGMENT_CHARS:
            return []
        if before == after:
            continue
        out.append(EditCandidate(
            before, after, confidence, category, "bounded token diff",
            start=offset + start,
        ))
    return out
