from __future__ import annotations

from dataclasses import dataclass
import difflib
import json
import re
from typing import Any


@dataclass(frozen=True)
class EditCandidate:
    """Одна предлагаемая правка.

    `start` — абсолютное смещение `before` в исходном тексте, если
    генератор кандидата его знает. До v9 позиция не передавалась, и
    `DecisionEngine` искал `before` подстрокой: любая правка неуникального
    слова («в», «раздел», «и») молча отбрасывалась. Смещение снимает это
    ограничение и убирает потери recall.

    `sources` — стадии, независимо предложившие ровно такую правку.
    Используется для голосования в `verification.CandidateArbiter`.
    """

    before: str
    after: str
    confidence: float = 1.0
    category: str = "unknown"
    reason: str = ""
    start: int | None = None
    sources: tuple[str, ...] = ()

    @property
    def votes(self) -> int:
        return max(1, len(self.sources))


class DecisionEngine:
    """Conservative merger between LLM edit candidates and deterministic guards."""

    def __init__(self, min_confidence: float = 0.55, max_changes: int = 40,
                 max_before_chars: int = 180, protected_words: set[str] | None = None,
                 guard: Any | None = None) -> None:
        self.min_confidence = min_confidence
        self.max_changes = max_changes
        self.max_before_chars = max_before_chars
        self.protected_words = {w.casefold() for w in (protected_words or set()) if w}
        self.guard = guard
        self.rejections: list[tuple[EditCandidate, str]] = []

    @staticmethod
    def parse(payload: str | dict[str, Any]) -> list[EditCandidate]:
        data = payload if isinstance(payload, dict) else json.loads(payload.strip().strip("`"))
        raw = data.get("edits", []) if isinstance(data, dict) else []
        if not isinstance(raw, list):
            return []
        out: list[EditCandidate] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            before, after = item.get("before"), item.get("after")
            if not isinstance(before, str) or not isinstance(after, str):
                continue
            try:
                confidence = float(item.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 0.0
            out.append(EditCandidate(
                before, after, max(0.0, min(1.0, confidence)),
                str(item.get("category", "unknown")), str(item.get("reason", "")),
            ))
        return out

    def _protected(self, before: str) -> bool:
        if not self.protected_words:
            return False
        tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9_-]*", before)
        return any(t.casefold() in self.protected_words for t in tokens)

    @staticmethod
    def _changes_compound_term(before: str, after: str) -> bool:
        """Hyphenated domain terms must not be replaced by a different lexeme."""
        before_terms = re.findall(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)+", before)
        after_terms = re.findall(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)+", after)
        return bool(before_terms and before_terms != after_terms)

    @staticmethod
    def _is_unverified_llm_inflection(c: EditCandidate) -> bool:
        """Reject model-only case/number substitutions of valid words.

        A generative model cannot establish that a heading such as
        ``Горючее`` must become genitive ``Горючего``. Such edits need a
        deterministic syntax signal; otherwise they are often stylistic
        hallucinations rather than corrections.
        """
        if not c.category.startswith(("model", "unknown", "languagetool", "surface")):
            return False
        if not re.fullmatch(r"[А-Яа-яЁё-]+", c.before) or not re.fullmatch(r"[А-Яа-яЁё-]+", c.after):
            return False
        try:
            from morphology import get_morphology

            morph = get_morphology()
            if not morph.available:
                return False
            before = morph.known_parses(c.before)
            after = morph.known_parses(c.after)
            if not before or not after:
                return False
            before_forms = {(p.normal_form, str(p.tag).split(",", 1)[0]) for p in before}
            after_forms = {(p.normal_form, str(p.tag).split(",", 1)[0]) for p in after}
            return bool(before_forms & after_forms)
        except Exception:
            return False

    @staticmethod
    def _splits_or_merges_word(c: EditCandidate) -> bool:
        """Reject model edits that change a word boundary without proof."""
        word = r"[А-Яа-яЁёA-Za-z]+"
        if re.fullmatch(word, c.before) and re.fullmatch(rf"{word} +{word}", c.after):
            return c.category.startswith(("model", "unknown", "languagetool", "surface"))
        if re.fullmatch(rf"{word} +{word}", c.before) and re.fullmatch(word, c.after):
            return c.category.startswith(("model", "unknown", "languagetool", "surface"))
        return False

    def validate(self, text: str, candidates: list[EditCandidate]) -> list[tuple[int, EditCandidate]]:
        accepted: list[tuple[int, EditCandidate]] = []
        occupied: list[tuple[int, int]] = []
        self.rejections = []

        def reject(candidate: EditCandidate, why: str) -> None:
            self.rejections.append((candidate, why))

        for c in sorted(candidates, key=lambda x: (-x.confidence, -len(x.before))):
            if len(accepted) >= self.max_changes or not c.before or c.before == c.after:
                continue
            if len(c.before) > self.max_before_chars or c.confidence < self.min_confidence:
                reject(c, "низкая уверенность или слишком длинный фрагмент")
                continue
            if c.before.replace("ё", "е").replace("Ё", "Е") == c.after.replace("ё", "е").replace("Ё", "Е"):
                reject(c, "различие только в ё/е")
                continue
            if self._protected(c.before):
                reject(c, "защищённый термин")
                continue
            if self._changes_compound_term(c.before, c.after):
                reject(c, "подмена составного термина")
                continue
            if self._is_unverified_llm_inflection(c):
                reject(c, "неподтверждённая падежная правка от модели")
                continue
            if self._splits_or_merges_word(c):
                reject(c, "изменение границы слова без доказательства")
                continue
            if self.guard is not None:
                allowed, why = self.guard.allow(c, text)
                if not allowed:
                    reject(c, why)
                    continue
            start = self._resolve_span(text, c)
            if start is None:
                reject(c, "фрагмент не найден однозначно")
                continue
            end = start + len(c.before)
            if any(not (end <= a or start >= b) for a, b in occupied):
                reject(c, "пересечение с принятой правкой")
                continue
            occupied.append((start, end))
            accepted.append((start, c))
        return sorted(accepted, key=lambda x: x[0], reverse=True)

    @staticmethod
    def _resolve_span(text: str, c: EditCandidate) -> int | None:
        """Определяет позицию правки: по смещению кандидата или поиском.

        Смещение, пришедшее от генератора, доверенное — оно вычислено по
        тому же тексту. Поиск подстрокой остаётся fallback-ом для
        кандидатов без позиции (LLM-JSON, внешние детекторы).
        """
        if c.start is not None and 0 <= c.start <= len(text) - len(c.before):
            if text[c.start:c.start + len(c.before)] == c.before:
                return c.start
        positions = [m.start() for m in re.finditer(re.escape(c.before), text)]
        if len(positions) != 1:
            return None
        return positions[0]

    def apply(self, text: str, candidates: list[EditCandidate]) -> tuple[str, list[EditCandidate]]:
        accepted_spans = self.validate(text, candidates)
        result = text
        for start, c in accepted_spans:
            result = result[:start] + c.after + result[start + len(c.before):]
        return result, [c for _, c in reversed(accepted_spans)]

    @staticmethod
    def diff_candidates(original: str, corrected: str) -> list[EditCandidate]:
        if original == corrected:
            return []
        sm = difflib.SequenceMatcher(a=original, b=corrected, autojunk=False)
        return [EditCandidate(original[i1:i2], corrected[j1:j2], 1.0, "diff", "server diff")
                for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]
