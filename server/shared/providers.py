"""Multi-provider конфигурация для cloud-сервера (v2.2.3+).

Позволяет в `server/cloud/.env` декларативно описать список любых
OpenAI-compatible провайдеров (OpenRouter, DeepSeek, Fireworks, Groq,
Together, OpenAI и т. д.) и для каждого — отдельный API-ключ, base URL
и список моделей.

Схема env:
    CLOUD_PROVIDERS=openrouter,deepseek,fireworks

    OPENROUTER_API_KEY=sk-or-v1-...
    OPENROUTER_BASE_URL=https://openrouter.ai/api/v1   # опц. (есть дефолт)
    OPENROUTER_MODELS=openrouter/free,qwen/qwen3-next-80b-a3b-instruct:free

    DEEPSEEK_API_KEY=sk-...
    DEEPSEEK_BASE_URL=https://api.deepseek.com/v1     # опц. (есть дефолт)
    DEEPSEEK_MODELS=deepseek-chat,deepseek-reasoner

    FIREWORKS_API_KEY=fw_...
    FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
    FIREWORKS_MODELS=accounts/fireworks/models/llama-v3p3-70b-instruct

Любой провайдер без ключа автоматически пропускается с предупреждением
в лог. Если все провайдеры пропущены — cloud-сервер всё равно стартует,
но `/suggest` будет отдавать 503 «no providers configured».

Back-compat:
- Если `CLOUD_PROVIDERS` не задан → дефолт `openrouter` (текущее поведение).
- Если для openrouter не задан `OPENROUTER_MODELS` → fallback на
  `CLOUD_PRESET` (A/B/C/D) и `CLOUD_PRESETS` из openrouter_client.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

from shared.openrouter_client import (
    OpenAICompatibleClient,
    resolve_models,
)


# ─── Известные провайдеры с дефолтным base_url ─────────────────────────


# Для популярных провайдеров можно не указывать <NAME>_BASE_URL —
# берётся дефолт. Все эндпоинты OpenAI-compatible: `/chat/completions`.
KNOWN_PROVIDERS: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "openai": "https://api.openai.com/v1",
    "mistral": "https://api.mistral.ai/v1",
}


def _normalize_env_prefix(name: str) -> str:
    """Преобразует имя провайдера в env-prefix.

    Примеры:
        openrouter      → OPENROUTER
        deepseek        → DEEPSEEK
        my-provider     → MY_PROVIDER
        with space      → WITH_SPACE
    """
    return name.upper().replace("-", "_").replace(" ", "_")


@dataclass
class ProviderConfig:
    """Конфигурация одного OpenAI-compatible провайдера.

    name:           идентификатор (например 'openrouter'), используется
                    как префикс для env-переменных.
    base_url:       URL без хвоста (без `/chat/completions`).
    api_key:        Bearer-токен.
    models:         список model id для перебора с fallback.
    extra_headers:  дополнительные HTTP-заголовки (например, для
                    OpenRouter: HTTP-Referer, X-Title).
    """

    name: str
    base_url: str
    api_key: str
    models: list[str] = field(default_factory=list)
    extra_headers: dict[str, str] = field(default_factory=dict)

    @property
    def key_present(self) -> bool:
        """True, если ключ заполнен и не выглядит как placeholder."""
        if not self.api_key:
            return False
        lower = self.api_key.lower()
        return "ваш_ключ" not in lower and "your_key" not in lower

    def build_client(
        self, *, timeout: float = 90.0, transport=None,
    ) -> OpenAICompatibleClient:
        return OpenAICompatibleClient(
            self.api_key,
            base_url=self.base_url,
            extra_headers=self.extra_headers,
            provider_name=self.name,
            timeout=timeout,
            transport=transport,
        )


def _load_one_provider(
    name: str,
    env_getter: Callable[[str], Optional[str]],
) -> ProviderConfig:
    """Читает один провайдер из env (по префиксу). Не валидирует ключ —
    верхний уровень фильтрует провайдеров без ключа."""
    prefix = _normalize_env_prefix(name)
    api_key = (env_getter(f"{prefix}_API_KEY") or "").strip()
    base_url = (env_getter(f"{prefix}_BASE_URL") or "").strip()
    if not base_url:
        base_url = KNOWN_PROVIDERS.get(name.lower(), "")

    models_csv = (env_getter(f"{prefix}_MODELS") or "").strip()
    models = [m.strip() for m in models_csv.split(",") if m.strip()]

    # OpenRouter: если MODELS не указан — берём CLOUD_PRESET (back-compat).
    if not models and name.lower() == "openrouter":
        preset = (env_getter("CLOUD_PRESET") or "A").strip().upper()
        models = resolve_models(preset)

    extra_headers: dict[str, str] = {}
    if name.lower() == "openrouter":
        # OpenRouter использует HTTP-Referer и X-Title для квот и атрибуции.
        referer = (env_getter("OPENROUTER_REFERER") or "http://localhost").strip()
        title = (env_getter("OPENROUTER_TITLE") or "AI LibreOffice Suggester").strip()
        extra_headers["HTTP-Referer"] = referer
        extra_headers["X-Title"] = title

    return ProviderConfig(
        name=name,
        base_url=base_url,
        api_key=api_key,
        models=models,
        extra_headers=extra_headers,
    )


def load_providers_from_env(
    env_getter: Optional[Callable[[str], Optional[str]]] = None,
) -> list[ProviderConfig]:
    """Загружает список провайдеров из env-переменных.

    `env_getter` — обычно `os.getenv` (по умолчанию). Для тестов можно
    передать собственный (dict.get) чтобы не мутировать процессное env.

    Возвращает все провайдеры из CLOUD_PROVIDERS в исходном порядке,
    включая те, у которых не задан ключ (фильтрацию делает верхний
    уровень, чтобы можно было логировать почему провайдер пропущен).
    """
    getter = env_getter if env_getter is not None else os.getenv

    raw = (getter("CLOUD_PROVIDERS") or "openrouter").strip()
    names = [s.strip() for s in raw.split(",") if s.strip()]
    if not names:
        names = ["openrouter"]

    seen: set[str] = set()
    providers: list[ProviderConfig] = []
    for name in names:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        providers.append(_load_one_provider(name, getter))
    return providers
