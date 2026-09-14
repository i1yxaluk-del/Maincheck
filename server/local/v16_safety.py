"""Защита от разрушительных правок генеративных моделей."""
from __future__ import annotations

from dataclasses import replace
import re

from decision_engine import EditCandidate
from morphology import features, get_morphology
from verification import is_generative

QUOTE_CHARS = frozenset('«»"“”„‟')
SOLO_PUNCTUATION_CONFIDENCE = 0.49
ABBREVIATION_RE = re.compile(r"(?<![А-Яа-яЁёA-Za-z])[А-ЯЁA-Z]{2,}(?![А-Яа-яЁёA-Za-z])")
DIGIT_TOKEN_RE = re.compile(r"(?<!\w)\d+(?!\w)")
WORD_RE = re.compile(r"[А-Яа-яЁё]+")


def _quote_signature(value: str) -> tuple[str, ...]:
    return tuple(ch for ch in value if ch in QUOTE_CHARS)


def _loses_protected_token(before: str, after: str) -> bool:
    """Не разрешает модели удалять числа и служебные аббревиатуры."""
    protected = ABBREVIATION_RE.findall(before) + DIGIT_TOKEN_RE.findall(before)
    return any(token not in after for token in protected)


def _fuses_words(before: str, after: str) -> bool:
    """Обнаруживает склейку нескольких исходных слов в один токен."""
    before_words = WORD_RE.findall(before)
    after_words = WORD_RE.findall(after)
    return len(before_words) >= 2 and len(after_words) == 1 and bool(re.search(r"\s", before))


def _breaks_parallel_participles(text: str, candidate: EditCandidate) -> bool:
    """Защищает согласование повторяющихся причастных характеристик лица."""
    if candidate.start is None or not candidate.before or not candidate.after:
        return False
    morph = get_morphology()
    before_parses = [p for p in morph.attributive_parses(candidate.before) if p.tag.POS == "PRTF"]
    after_parses = [p for p in morph.attributive_parses(candidate.after) if p.tag.POS == "PRTF"]
    if not before_parses or not after_parses:
        return False
    tail_start = candidate.start + len(candidate.before)
    tail = text[tail_start:tail_start + 240]
    for match in re.finditer(rf",\s*(?P<word>{WORD_RE.pattern})\b", tail):
        peer_parses = [
            p for p in morph.attributive_parses(match.group("word"))
            if p.tag.POS == "PRTF"
        ]
        if not peer_parses:
            continue
        was_parallel = any(
            features(left).agrees_with(features(peer))
            for left in before_parses for peer in peer_parses
        )
        remains_parallel = any(
            features(right).agrees_with(features(peer))
            for right in after_parses for peer in peer_parses
        )
        if was_parallel and not remains_parallel:
            return True
    return False


def suppress_unsafe_solo_punctuation(
    candidates: list[EditCandidate], text: str = "",
) -> list[EditCandidate]:
    """Понижает небезопасные модельные правки ниже порога принятия.

    Кроме пунктуации защищает числа, аббревиатуры, границы слов и
    согласованные параллельные причастия. Детерминированные правила этими
    ограничениями не затрагиваются.
    """
    out: list[EditCandidate] = []
    for candidate in candidates:
        if not is_generative(candidate.category):
            out.append(candidate)
            continue
        sources = candidate.sources or (candidate.category,)
        has_rupunct = any("rupunct" in source for source in sources)
        risky_quote = _quote_signature(candidate.before) != _quote_signature(candidate.after)
        solo_comma_addition = (
            not has_rupunct
            and len(sources) == 1
            and candidate.after.count(",") > candidate.before.count(",")
        )
        protected_loss = _loses_protected_token(candidate.before, candidate.after)
        fused_words = _fuses_words(candidate.before, candidate.after)
        broken_parallelism = bool(text) and _breaks_parallel_participles(text, candidate)
        if risky_quote or solo_comma_addition or protected_loss or fused_words or broken_parallelism:
            out.append(replace(
                candidate,
                confidence=min(candidate.confidence, SOLO_PUNCTUATION_CONFIDENCE),
            ))
        else:
            out.append(candidate)
    return out
