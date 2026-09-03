from __future__ import annotations

import difflib
import logging
import os
import re
from dataclasses import dataclass

import httpx

logger = logging.getLogger("ai_suggester.secondary_gec")

WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+(?:[-/][А-Яа-яЁёA-Za-z0-9]+)*")
MODEL_DEFAULT = "hf.co/loqira/Qwen3.5-0.8B-GEC-KAZ-RUS-ENG:Q4_0"
# The model card explicitly recommends this exact Russian prompt for best quality.
SYSTEM = (
    "Отформатируй текст голосового ввода: расставь пунктуацию и заглавные "
    "буквы, разбей на абзацы, исправь опечатки и орфографические ошибки. "
    "Сохрани язык, слова и смысл, ничего не добавляй от себя."
)


@dataclass(frozen=True)
class SecondaryEdit:
    before: str
    after: str
    category: str


def _tokens(text: str) -> list[str]:
    return [m.group(0) for m in WORD_RE.finditer(text)]


def _known(morph, word: str) -> bool:
    try:
        return bool(morph.word_is_known(word)) if morph else False
    except Exception:
        return False


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def merge_safe(
    primary: str,
    secondary: str,
    morph,
    max_edits: int = 4,
) -> tuple[str, list[SecondaryEdit]]:
    """Merge only conservative surface edits from secondary into primary.

    The secondary model is intentionally not allowed to replace valid inflections,
    rewrite lexical choices, or flood the client with a large speculative diff.
    """
    if not secondary or secondary.strip() == primary.strip():
        return primary, []

    sm = difflib.SequenceMatcher(None, primary, secondary, autojunk=False)
    accepted = []
    edits: list[SecondaryEdit] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        before = primary[i1:i2]
        after = secondary[j1:j2]
        before_words = _tokens(before)
        after_words = _tokens(after)

        # Pure punctuation/spacing/capitalization. One side may be empty because
        # punctuation can be inserted/deleted without changing lexical tokens.
        if before_words == after_words:
            accepted.append((i1, i2, after))
            edits.append(SecondaryEdit(before, after, "punctuation/typography"))
            continue

        # Never accept a lexical deletion/insertion from the secondary model.
        if not before or not after:
            continue

        # Conservative spelling correction: only one token, source unknown,
        # target known, and small edit distance. This rejects valid inflections.
        if len(before_words) == len(after_words) == 1:
            src, dst = before_words[0], after_words[0]
            if not _known(morph, src) and _known(morph, dst):
                limit = 1 if max(len(src), len(dst)) <= 6 else 2
                if _edit_distance(src.lower(), dst.lower()) <= limit:
                    accepted.append((i1, i2, after))
                    edits.append(SecondaryEdit(src, dst, "spelling"))

    if not accepted:
        return primary, []
    if len(accepted) > max_edits:
        logger.info(
            "SecondaryGEC: reject secondary output, accepted candidates=%d > max_edits=%d",
            len(accepted),
            max_edits,
        )
        return primary, []

    out = []
    cursor = 0
    for i1, i2, after in accepted:
        if i1 < cursor:
            continue
        out.extend((primary[cursor:i1], after))
        cursor = i2
    out.append(primary[cursor:])
    return "".join(out), edits


class SecondaryGEC:
    def __init__(
        self,
        model: str,
        timeout: float,
        keep_alive: str,
        max_edits: int = 4,
    ) -> None:
        self.enabled = True
        self.model = model
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.max_edits = max(1, max_edits)
        self.base_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self._morph = None
        try:
            import pymorphy3

            self._morph = pymorphy3.MorphAnalyzer()
        except Exception as exc:
            logger.warning("SecondaryGEC: pymorphy3 unavailable: %s", exc)

    async def check_available(self) -> bool:
        """Return whether the configured secondary model is present in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=min(self.timeout, 10.0)) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
            names = {str(item.get("name", "")) for item in models}
            available = self.model in names
            if available:
                logger.info("SecondaryGEC ready: model=%s найден в Ollama", self.model)
            else:
                logger.warning(
                    "SecondaryGEC model=%s не найден в Ollama. "
                    "Для preset C выполните: ollama pull %s",
                    self.model,
                    self.model,
                )
            return available
        except Exception as exc:
            logger.warning("SecondaryGEC availability check failed: %s", exc)
            return False

    async def enrich(self, primary_text: str) -> tuple[str, list[SecondaryEdit]]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": primary_text},
            ],
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0,
                "top_k": 1,
                "repeat_penalty": 1.0,
                "num_ctx": 2048,
                "num_predict": 512,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                secondary = response.json().get("message", {}).get("content", "").strip()
            merged, edits = merge_safe(
                primary_text,
                secondary,
                self._morph,
                max_edits=self.max_edits,
            )
            logger.info(
                "SecondaryGEC: model=%s accepted=%d max_edits=%d",
                self.model,
                len(edits),
                self.max_edits,
            )
            return merged, edits
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.warning(
                    "SecondaryGEC: model=%s не загружена в Ollama (HTTP 404); "
                    "выполните `ollama pull %s`",
                    self.model,
                    self.model,
                )
            else:
                logger.warning("SecondaryGEC failed, keeping primary result: %s", exc)
            return primary_text, []
        except Exception as exc:
            logger.warning("SecondaryGEC failed, keeping primary result: %s", exc)
            return primary_text, []
