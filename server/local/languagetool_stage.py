"""Стадия LanguageTool в локальном пайплайне (v9).

Почему это важно
================
Жалоба пользователя включает пунктуацию и стилистику. Ни SAGE-95m, ни
Qwen3.5-0.8B-GEC на этих классах не надёжны, а генеративные правки
пунктуации невозможно верифицировать морфологией. LanguageTool — это
900+ *детерминированных* правил для русского языка (LGPL, полностью
офлайн), которые дают проверяемое объяснение каждой правки.

Клиент `shared/languagetool_client.py` и переменные `LANGUAGETOOL_*` в
`.env.example` существовали в репозитории с v2.0, но в v8-путь
(`decision_app` → `hybrid_editor`) подключены не были. Эта стадия
закрывает разрыв.

Развёртывание (офлайн, порт 8081)::

    docker run -d --restart=always -p 8081:8010 \\
        -e langtool_languageModel=/ngrams \\
        -e Java_Xmx=2g --name languagetool erikvl87/languagetool

Публичный api.languagetool.org использовать запрещено: служебные
документы не должны покидать сервер. Клиент по умолчанию ходит только на
127.0.0.1.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

from decision_engine import EditCandidate

logger = logging.getLogger("ai_suggester.languagetool")

#: Категории, где LanguageTool даёт наибольшую пользу и не дублирует
#: детерминированные правила сервера. GRAMMAR и TYPOS по умолчанию
#: выключены: их закрывают `local_rules` и `spellcheck`, причём с
#: морфологическим доказательством.
DEFAULT_CATEGORIES = "PUNCTUATION,TYPOGRAPHY"


@dataclass(frozen=True)
class LanguageToolStats:
    enabled: bool
    url: str
    calls: int
    matches: int


class LanguageToolStage:
    def __init__(self) -> None:
        self.enabled = os.getenv("LANGUAGETOOL_ENABLED", "false").lower() in {
            "1", "true", "yes", "on",
        }
        self.url = os.getenv("LANGUAGETOOL_URL", "http://127.0.0.1:8081")
        self.confidence = float(os.getenv("LANGUAGETOOL_CONFIDENCE", "0.90"))
        self.max_matches = int(os.getenv("LANGUAGETOOL_MAX_MATCHES", "12"))
        self._calls = 0
        self._matches = 0
        self._client = None
        if not self.enabled:
            return
        try:
            from shared.languagetool_client import LanguageToolClient, _parse_csv_env

            self._client = LanguageToolClient(
                url=self.url,
                language=os.getenv("LANGUAGETOOL_LANGUAGE", "ru-RU"),
                enabled_categories=_parse_csv_env(
                    os.getenv("LANGUAGETOOL_ENABLED_CATEGORIES", DEFAULT_CATEGORIES)
                ),
                disabled_categories=_parse_csv_env(os.getenv("LANGUAGETOOL_DISABLED_CATEGORIES")),
                disabled_rules=_parse_csv_env(os.getenv("LANGUAGETOOL_DISABLED_RULES")),
                timeout=float(os.getenv("LANGUAGETOOL_TIMEOUT", "10")),
            )
        except Exception as exc:
            logger.warning("LanguageTool stage unavailable: %s", exc)
            self._client = None

    @property
    def available(self) -> bool:
        return self.enabled and self._client is not None

    async def candidates(self, text: str) -> list[EditCandidate]:
        if not self.available or not text.strip():
            return []
        self._calls += 1
        matches = await asyncio.to_thread(self._check, text)
        out: list[EditCandidate] = []
        for match in matches[: self.max_matches]:
            before = getattr(match, "before", "")
            after = getattr(match, "suggestion", "")
            if not before or not after or before == after:
                continue
            category = str(getattr(match, "category_id", "") or "generic").lower()
            out.append(EditCandidate(
                before, after, self.confidence,
                f"languagetool-{category}",
                f"LanguageTool {getattr(match, 'rule_id', '')}: {getattr(match, 'message', '')}".strip(),
                start=getattr(match, "offset", None),
            ))
        self._matches += len(out)
        return out

    def _check(self, text: str) -> list:
        try:
            return list(self._client.check(text))
        except Exception as exc:  # pragma: no cover - сетевой путь
            logger.warning("LanguageTool check failed: %s", exc)
            return []

    async def warmup(self) -> None:
        if self.available:
            await self.candidates("Проверка запуска корректора.")

    def metrics(self) -> LanguageToolStats:
        return LanguageToolStats(self.enabled, self.url, self._calls, self._matches)
