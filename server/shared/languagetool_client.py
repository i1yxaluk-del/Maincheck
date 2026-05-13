"""v2.0-b: клиент LanguageTool-RU HTTP-сервера.

Назначение: параллельный детектор стилистических и типографских правок
поверх T-lite/MorphDetector. LanguageTool (LGPL, Java) — open-source
GEC-движок с 930+ rule-based проверками для русского языка
(см. https://community.languagetool.org/rule/list?lang=ru).

Категории и их типичные правила:
  * STYLE       — стилистические подсказки (повторы, канцеляризмы)
  * TYPOGRAPHY  — типографика (тире вместо дефиса, кавычки «…»)
  * GRAMMAR     — грамматика (T-lite/MorphDetector уже хорошо ловят, обычно
                  отключают для избежания дублирования)
  * TYPOS       — опечатки (так же дублирует T-lite)
  * REDUNDANCY  — избыточность
  * PUNCTUATION — пунктуация

Архитектура:
  * Запускается отдельный LanguageTool-сервер (Docker / Java jar /
    нативный пакет), порт по умолчанию 8081
  * Этот клиент стучит в POST /v2/check с языком ru-RU
  * Если сервер недоступен — `available` становится False, вызовы
    `check()` возвращают пустой список (graceful fallback, не падаем)

Latency:
  * Локальный LT-сервер на типовом тексте 700 chars: ~50-200 мс
  * Cold-start LT-сервера (загрузка моделей): ~15-30 с

Безопасность:
  * Не отправлять текст на api.languagetool.org (публичный SaaS, не GDPR-safe)
  * Использовать только локальный self-hosted сервер
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import FrozenSet, Optional

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore
    _HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LTMatch:
    """Одна правка от LanguageTool.

    Поля совпадают по семантике с `morph_detector.GrammarError`, чтобы
    интеграция в pipeline была единообразной.
    """
    offset: int                # char offset в исходном тексте
    length: int                # длина исходного фрагмента
    before: str                # исходный фрагмент (текст[offset:offset+length])
    suggestion: str            # первая рекомендуемая замена (может быть пустой)
    message: str               # человекочитаемое объяснение LT
    category_id: str           # CATEGORY_ID (STYLE, TYPOGRAPHY, GRAMMAR, ...)
    rule_id: str               # ID правила LT (для disabledRules)

    def to_change_line(self, number: int) -> str:
        """Форматирует в строку CHANGES блока.

        Формат соответствует существующему `GrammarError.to_change_line`:
        `N. «before» → «suggestion» | message`.

        Категория добавляется в message для прозрачности.
        """
        cat = f"[{self.category_id}] " if self.category_id else ""
        # Не выходим за пределы 1 строки. Длинные LT-message обрезаем.
        msg = self.message.strip().replace("\n", " ")
        if len(msg) > 140:
            msg = msg[:137] + "…"
        return (
            f"{number}. «{self.before}» → «{self.suggestion}» | {cat}{msg}"
        )


def _parse_csv_env(value: Optional[str]) -> FrozenSet[str]:
    """Парсит CSV из env-переменной в frozenset. Пустые элементы пропускаются."""
    if not value:
        return frozenset()
    return frozenset(
        item.strip().upper() for item in value.split(",") if item.strip()
    )


class LanguageToolClient:
    """HTTP-клиент для LanguageTool-сервера.

    Lifecycle:
        client = LanguageToolClient(url="http://localhost:8081")
        if client.available:
            matches = client.check("Текст с ошибками.")

    Все вызовы non-throwing: при ошибке сети/таймауте/неверном ответе
    логируют warning и возвращают [].
    """

    def __init__(
        self,
        url: str = "http://localhost:8081",
        language: str = "ru-RU",
        enabled_categories: FrozenSet[str] = frozenset(),
        disabled_categories: FrozenSet[str] = frozenset(),
        disabled_rules: FrozenSet[str] = frozenset(),
        timeout: float = 10.0,
        transport=None,
    ):
        self.url = url.rstrip("/")
        self.language = language
        self.enabled_categories = enabled_categories
        self.disabled_categories = disabled_categories
        self.disabled_rules = disabled_rules
        self.timeout = timeout
        # Опциональный httpx.MockTransport (или другой). Используется в
        # тестах, чтобы не патчить httpx глобально (это ломает
        # starlette.TestClient). В проде = None → создаётся реальный сокет.
        self._transport = transport
        self._available: Optional[bool] = None

    def _make_client(self, timeout: float):
        """Создаёт httpx.Client с опциональным transport-override."""
        kwargs: dict = {"timeout": timeout}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.Client(**kwargs)

    @property
    def available(self) -> bool:
        """Лениво проверяет доступность LT-сервера. Кэшируется навсегда:
        если LT-сервер перезапустить — нужно создать новый клиент или
        вызвать `reset_availability()`.
        """
        if self._available is None:
            self._available = self._probe()
        return self._available

    def reset_availability(self) -> None:
        """Сбрасывает кеш availability — при следующем `available`
        будет новый probe-запрос."""
        self._available = None

    def _probe(self) -> bool:
        """Один запрос GET /v2/languages для проверки что сервер жив."""
        if not _HTTPX_AVAILABLE:
            logger.info("LanguageToolClient: httpx не установлен, недоступно")
            return False
        try:
            with self._make_client(timeout=min(3.0, self.timeout)) as c:
                r = c.get(f"{self.url}/v2/languages")
            if r.status_code != 200:
                logger.info(
                    "LanguageToolClient: GET /v2/languages вернул %d, недоступно",
                    r.status_code,
                )
                return False
            # Дополнительно проверим что наш language есть в списке
            langs = r.json()
            if not any(
                lang.get("longCode") == self.language
                or lang.get("code") == self.language.split("-")[0]
                for lang in langs
            ):
                logger.warning(
                    "LanguageToolClient: language=%s не найден в /v2/languages",
                    self.language,
                )
                return False
            return True
        except Exception as exc:  # noqa: BLE001
            logger.info("LanguageToolClient: probe failed (%s)", exc)
            return False

    def check(self, text: str) -> list[LTMatch]:
        """Запрос POST /v2/check, возвращает список LTMatch.

        При любой ошибке (сеть, таймаут, не-JSON, статус != 200) возвращает
        пустой список и логирует warning.
        """
        if not _HTTPX_AVAILABLE:
            return []
        if not text or not text.strip():
            return []
        params: dict[str, str] = {
            "text": text,
            "language": self.language,
        }
        if self.enabled_categories:
            params["enabledCategories"] = ",".join(self.enabled_categories)
            params["enabledOnly"] = "false"
        if self.disabled_categories:
            params["disabledCategories"] = ",".join(self.disabled_categories)
        if self.disabled_rules:
            params["disabledRules"] = ",".join(self.disabled_rules)
        try:
            with self._make_client(timeout=self.timeout) as c:
                r = c.post(f"{self.url}/v2/check", data=params)
            if r.status_code != 200:
                logger.warning(
                    "LanguageTool /v2/check вернул %d: %s",
                    r.status_code, r.text[:200],
                )
                return []
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("LanguageTool /v2/check failed: %s", exc)
            return []
        return [
            self._parse_match(m, text)
            for m in data.get("matches", [])
            if self._match_is_useful(m)
        ]

    def _match_is_useful(self, m: dict) -> bool:
        """True если matche имеет offset + length + хотя бы 1 replacement
        ИЛИ это категория INFO (стилистическая подсказка без замены).
        Сейчас пропускаем матчи без replacements — пользователю нужна
        чёткая правка, а не «обратите внимание».
        """
        offset = m.get("offset")
        length = m.get("length")
        replacements = m.get("replacements", [])
        if offset is None or length is None or length <= 0:
            return False
        if not replacements:
            return False
        first = replacements[0].get("value", "")
        if not first:
            return False
        return True

    def _parse_match(self, m: dict, source_text: str) -> LTMatch:
        offset = int(m.get("offset", 0))
        length = int(m.get("length", 0))
        before = source_text[offset:offset + length]
        replacements = m.get("replacements", [])
        suggestion = replacements[0].get("value", "") if replacements else ""
        rule = m.get("rule", {}) or {}
        category = rule.get("category", {}) or {}
        return LTMatch(
            offset=offset,
            length=length,
            before=before,
            suggestion=suggestion,
            message=m.get("message", ""),
            category_id=str(category.get("id", "")).upper(),
            rule_id=str(rule.get("id", "")),
        )


# ─── Singleton helper ──────────────────────────────────────────────────

_client: Optional[LanguageToolClient] = None


def get_languagetool_client(
    url: str = "http://localhost:8081",
    language: str = "ru-RU",
    enabled_categories: FrozenSet[str] = frozenset(),
    disabled_categories: FrozenSet[str] = frozenset(),
    disabled_rules: FrozenSet[str] = frozenset(),
    timeout: float = 10.0,
    transport=None,
) -> LanguageToolClient:
    """Возвращает синглтон. Повторные вызовы с разными параметрами НЕ
    создают новый объект — для смены конфига вызовите `reset_client()`."""
    global _client
    if _client is None:
        _client = LanguageToolClient(
            url=url,
            language=language,
            enabled_categories=enabled_categories,
            disabled_categories=disabled_categories,
            disabled_rules=disabled_rules,
            timeout=timeout,
            transport=transport,
        )
    return _client


def reset_client() -> None:
    """Сбрасывает синглтон. Используется в тестах и при смене конфига."""
    global _client
    _client = None


__all__ = [
    "LTMatch",
    "LanguageToolClient",
    "get_languagetool_client",
    "reset_client",
    "_parse_csv_env",
]
