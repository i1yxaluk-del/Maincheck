from __future__ import annotations

import difflib
import logging
import os
import re

import httpx

logger = logging.getLogger("ai_suggester.secondary_gec")

WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+(?:[-/][А-Яа-яЁёA-Za-z0-9]+)*")
MODEL_DEFAULT = "hf.co/loqira/Qwen3.5-0.8B-GEC-KAZ-RUS-ENG:Q4_0"
SYSTEM = (
    "Ты корректор русского текста. Исправляй только явные орфографические, "
    "пунктуационные и типографические ошибки. Не меняй грамматику, управление, "
    "смысл, лексику или допустимые формы слов. Верни только исправленный текст."
)


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


def merge_safe(primary: str, secondary: str, morph) -> tuple[str, int]:
    if not secondary or secondary.strip() == primary.strip():
        return primary, 0
    sm = difflib.SequenceMatcher(None, primary, secondary, autojunk=False)
    accepted = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        before = primary[i1:i2]
        after = secondary[j1:j2]
        if not before or not after:
            continue
        before_words = _tokens(before)
        after_words = _tokens(after)
        # Pure punctuation/spacing/capitalization: lexical tokens unchanged.
        if before_words == after_words:
            accepted.append((i1, i2, after))
            continue
        # Spelling only: unknown -> known, small edit distance. Inflectional
        # changes such as наряда -> нарядов remain rejected because both forms
        # are valid dictionary words.
        if len(before_words) == len(after_words) == 1:
            src, dst = before_words[0], after_words[0]
            if not _known(morph, src) and _known(morph, dst):
                limit = 1 if max(len(src), len(dst)) <= 6 else 2
                if _edit_distance(src.lower(), dst.lower()) <= limit:
                    accepted.append((i1, i2, after))
    if not accepted:
        return primary, 0
    out = []
    cursor = 0
    for i1, i2, after in accepted:
        if i1 < cursor:
            continue
        out.extend((primary[cursor:i1], after))
        cursor = i2
    out.append(primary[cursor:])
    return "".join(out), len(accepted)


class SecondaryGEC:
    def __init__(self) -> None:
        self.enabled = os.getenv("SECONDARY_GEC_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
        self.model = os.getenv("SECONDARY_GEC_MODEL", MODEL_DEFAULT)
        self.timeout = float(os.getenv("SECONDARY_GEC_TIMEOUT", "90"))
        self.keep_alive = os.getenv("SECONDARY_GEC_KEEP_ALIVE", "5m")
        self.base_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self._morph = None
        if self.enabled:
            try:
                import pymorphy3
                self._morph = pymorphy3.MorphAnalyzer()
            except Exception as exc:
                logger.warning("SecondaryGEC: pymorphy3 unavailable: %s", exc)

    async def enrich(self, primary_text: str) -> str:
        if not self.enabled:
            return primary_text
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
                "num_ctx": 2048,
                "num_predict": 512,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                secondary = response.json().get("message", {}).get("content", "").strip()
            merged, accepted = merge_safe(primary_text, secondary, self._morph)
            logger.info("SecondaryGEC: model=%s accepted=%d", self.model, accepted)
            return merged
        except Exception as exc:
            logger.warning("SecondaryGEC failed, keeping primary result: %s", exc)
            return primary_text
