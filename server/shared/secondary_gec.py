"""Secondary surface GEC pass for spelling/punctuation/capitalization.

The primary GEC model is responsible for grammar and semantic/contextual edits.
This optional pass uses a compact proofreading model trained specifically for
spelling, punctuation, capitalization and paragraph restoration. Its output is
NOT trusted wholesale: only surface-safe edits are merged into the primary
corrected text.
"""
from __future__ import annotations

import difflib
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

import httpx

_LOG = logging.getLogger("ai_suggester.secondary_gec")

DEFAULT_MODEL = "hf.co/loqira/Qwen3.5-0.8B-GEC-KAZ-RUS-ENG:Q4_0"
SYSTEM_PROMPT = (
    "Отформатируй текст голосового ввода: расставь пунктуацию и заглавные буквы, "
    "разбей на абзацы, исправь опечатки и орфографические ошибки. Сохрани язык, "
    "слова и смысл, ничего не добавляй от себя."
)

_WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+(?:[-/][А-Яа-яЁёA-Za-z0-9]+)*")


@dataclass(frozen=True)
class SurfaceEdit:
    before: str
    after: str
    kind: str


def _tokens(text: str) -> list[str]:
    return [m.group(0) for m in _WORD_RE.finditer(text)]


def _is_capitalization_only(before: str, after: str) -> bool:
    return before.lower() == after.lower() and before != after


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _word_known(morph, word: str) -> bool:
    if morph is None:
        return False
    try:
        return bool(morph.word_is_known(word))
    except Exception:
        return False


def _classify_change(before: str, after: str, morph) -> Optional[str]:
    """Return a conservative class that is safe to merge."""
    if before == after:
        return None
    if not before or not after:
        return None
    if _is_capitalization_only(before, after):
        return "capitalization"
    before_words = _tokens(before)
    after_words = _tokens(after)
    if before_words != after_words and len(before_words) == len(after_words) == 1:
        # Spelling gate: only allow an unknown source token becoming a known
        # token with a small edit distance. This prevents the small model from
        # overriding a valid grammatical choice such as наряда -> нарядов.
        if not _word_known(morph, before_words[0]) and _word_known(morph, after_words[0]):
            limit = 1 if max(len(before_words[0]), len(after_words[0])) <= 6 else 2
            if _levenshtein(before_words[0].lower(), after_words[0].lower()) <= limit:
                return "spelling"
        return None
    # Punctuation / whitespace edits preserve the lexical token sequence.
    if before_words == after_words:
        return "punctuation"
    return None


def _extract_corrected(text: str) -> str:
    if "===CORRECTED===" not in text or "===CHANGES===" not in text:
        return ""
    try:
        _, tail = text.split("===CORRECTED===", 1)
        body, _ = tail.split("===CHANGES===", 1)
    except ValueError:
        return ""
    return body.strip()


def _apply_safe_secondary(raw_text: str, primary_text: str, secondary_text: str, morph) -> tuple[str, list[SurfaceEdit]]:
    """Merge only surface-safe secondary edits into primary_text."""
    if not secondary_text or secondary_text.strip() == primary_text.strip():
        return primary_text, []
    sm = difflib.SequenceMatcher(None, primary_text, secondary_text, autojunk=False)
    accepted_ops: list[tuple[int, int, str]] = []
    edits: list[SurfaceEdit] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        before = primary_text[i1:i2]
        after = secondary_text[j1:j2]
        kind = _classify_change(before, after, morph)
        if kind is None:
            _LOG.info("Secondary GEC: отклонена небезопасная правка %r -> %r", before, after)
            continue
        accepted_ops.append((i1, i2, after))
        edits.append(SurfaceEdit(before=before, after=after, kind=kind))

    if not accepted_ops:
        return primary_text, []

    result_parts: list[str] = []
    cursor = 0
    for i1, i2, after in accepted_ops:
        if i1 < cursor:
            continue
        result_parts.append(primary_text[cursor:i1])
        result_parts.append(after)
        cursor = i2
    result_parts.append(primary_text[cursor:])
    merged = "".join(result_parts)
    return merged, edits


class SecondaryGEC:
    """Compact local proofreading pass using Ollama."""

    def __init__(
        self,
        base_url: str,
        model: str = DEFAULT_MODEL,
        timeout: float = 90.0,
        keep_alive: str = "5m",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.keep_alive = keep_alive
        self._morph = None
        try:
            import pymorphy3  # type: ignore[import-untyped]
            self._morph = pymorphy3.MorphAnalyzer()
        except Exception as exc:
            _LOG.warning("Secondary GEC: pymorphy3 недоступен для spelling-gate: %s", exc)

    async def correct(self, text: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0,
                "num_ctx": 2048,
                "num_predict": 512,
                "repeat_penalty": 1.0,
            },
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            result = response.json()["message"]["content"].strip()
        # The model is single-turn/plain-text; tolerate accidental wrappers.
        if "===CORRECTED===" in result:
            result = _extract_corrected(result)
        return result

    async def enrich(self, primary_text: str) -> tuple[str, list[SurfaceEdit]]:
        secondary = await self.correct(primary_text)
        merged, edits = _apply_safe_secondary(
            primary_text,
            primary_text,
            secondary,
            self._morph,
        )
        return merged, edits
