"""Политика эскалации каскада по полноте найденных исправлений."""
from __future__ import annotations

import os
import re

from decision_engine import EditCandidate

SKELETON_RE = re.compile(r"[А-Яа-яЁёA-Za-z]+|\d+")


def is_punctuation_only(candidate: EditCandidate) -> bool:
    """Возвращает True, если правка меняет только пунктуацию или пробелы."""
    return SKELETON_RE.findall(candidate.before) == SKELETON_RE.findall(candidate.after)


def needs_deep_review(text: str, candidates: list[EditCandidate]) -> bool:
    """Запускает углублённую проверку без содержательной языковой правки.

    Старая проверка ``if fast: stop`` считала одну найденную запятую
    доказательством проверки всего выделения. Теперь решение принимается
    по покрытию: пунктуационные кандидаты не отключают проверку грамматики.
    """
    min_chars = int(os.getenv("REASONING_COVERAGE_MIN_CHARS", "50"))
    if len(text.strip()) < min_chars:
        return not candidates
    return not candidates or all(is_punctuation_only(c) for c in candidates)


def needs_rescue_despite_verified_punctuation(
    candidates: list[EditCandidate], original_decision: bool,
) -> bool:
    """Не позволяет подтверждённой запятой отключить rescue в A/B/Y."""
    if candidates and all(is_punctuation_only(c) for c in candidates):
        return True
    return original_decision
