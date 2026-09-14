"""Структурное разбиение с жёстким пределом размера каждого фрагмента."""
from __future__ import annotations
import re
from typing import Iterable

_BOUNDARIES = ("\n\n", "\n", ". ", "; ", ", ", " ")


def safe_chunk_text(text: str, *, chunk_chars: int = 1200, overlap: int = 150) -> Iterable[str]:
    """Режет даже один огромный абзац RTF; ни один чанк не превышает лимит.

    Прежний алгоритм соблюдал лимит только между абзацами. Если экспортированный
    RTF превращался в один абзац, в Ollama уходил весь документ и модель
    эмбеддингов отвечала `input length exceeds the context length`.
    """
    if chunk_chars <= 0:
        raise ValueError("chunk_chars должен быть > 0")
    overlap = max(0, min(overlap, chunk_chars // 3))
    source = re.sub(r"[ \t]+", " ", text).strip()
    start = 0
    size = len(source)
    while start < size:
        hard_end = min(size, start + chunk_chars)
        end = hard_end
        if hard_end < size:
            floor = start + max(1, chunk_chars // 2)
            for marker in _BOUNDARIES:
                candidate = source.rfind(marker, floor, hard_end)
                if candidate >= floor:
                    end = candidate + (0 if marker == " " else len(marker))
                    break
        chunk = source[start:end].strip()
        if chunk:
            if len(chunk) > chunk_chars:
                raise AssertionError("внутренняя ошибка: превышен размер чанка")
            yield chunk
        if end >= size:
            break
        next_start = max(start + 1, end - overlap)
        # Не начинаем новый фрагмент посередине слова, если рядом есть пробел.
        space = source.find(" ", next_start, min(end, next_start + 80))
        start = space + 1 if space >= 0 else next_start
