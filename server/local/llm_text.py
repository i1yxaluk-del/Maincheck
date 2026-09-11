"""Нормализация ответов локальных LLM (v9).

Зачем
=====
v8 передавал ответ Ollama прямо в `safe_diff.diff_candidates`. Если
модель добавляла хоть одно служебное слово — «Исправленный текст:»,
``` ```-обёртку, `<think>`-блок, кавычки вокруг результата или
завершающий комментарий — diff получался глобальным,
`_global_rewrite` отбрасывал его целиком, и стадия молча давала ноль
кандидатов. В логах это выглядит как «модель ничего не находит».

Для инструктивных моделей квантованных в Q4 такие обёртки — норма, а не
исключение, поэтому очистка ответа обязательна, а не опциональна.

Дополнительно проверяем инвариант «модель вернула тот же текст, а не
пересказ»: если в ответе нет ни одного слова из исходного фрагмента или
длина изменилась больше чем на 35 %, ответ отбрасывается до diff-а с
явной причиной в метриках.
"""

from __future__ import annotations

import re

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_LEADING_LABEL = re.compile(
    r"^\s*(?:исправленн(?:ый|ая|ое)\s+(?:текст|вариант|версия)|"
    r"результат|ответ|итог|correct(?:ed)?\s*text|output)\s*[:\-—]\s*",
    re.I,
)
_TRAILING_NOTE = re.compile(
    r"\n\s*(?:пояснени|коммент|примечани|изменени|исправлени|note|explanation)\w*\s*[:\-—].*$",
    re.I | re.S,
)
_WORD = re.compile(r"[А-Яа-яЁёA-Za-z]{3,}")
_QUOTES = "\"'«»„“”‘’"


def normalize_llm_output(raw: str) -> str:
    """Снимает типовые обёртки инструктивных моделей."""
    if not raw:
        return ""
    text = _THINK.sub("", raw).strip()
    text = _FENCE.sub("", text).strip()
    text = _LEADING_LABEL.sub("", text).strip()
    text = _TRAILING_NOTE.sub("", text).strip()
    # Кавычки вокруг всего ответа снимаем только парой, чтобы не
    # тронуть кавычки внутри текста документа.
    while len(text) >= 2 and text[0] in _QUOTES and text[-1] in _QUOTES:
        stripped = text[1:-1].strip()
        if not stripped:
            break
        text = stripped
    return text


def looks_like_same_text(source: str, produced: str, max_length_drift: float = 0.35) -> bool:
    """Проверяет, что модель вернула исправленный фрагмент, а не пересказ."""
    if not produced:
        return False
    baseline = max(1, len(source))
    if abs(len(produced) - len(source)) / baseline > max_length_drift:
        return False
    source_words = {w.casefold() for w in _WORD.findall(source)}
    if not source_words:
        return True
    produced_words = {w.casefold() for w in _WORD.findall(produced)}
    overlap = len(source_words & produced_words) / len(source_words)
    return overlap >= 0.6


def sanitize(source: str, raw: str) -> str:
    """Нормализует ответ и возвращает пустую строку, если он непригоден."""
    produced = normalize_llm_output(raw)
    if not produced or produced == source:
        return ""
    return produced if looks_like_same_text(source, produced) else ""
