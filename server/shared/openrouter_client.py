"""HTTP-клиент для OpenAI-совместимых API с fallback-цепочкой моделей.

v2.2.3: обобщён `OpenRouterClient` → `OpenAICompatibleClient`. Любой
OpenAI-compatible эндпоинт (`/chat/completions`) работает через этот
клиент: OpenRouter, DeepSeek, Fireworks, Groq, Together, OpenAI, и т. д.
`OpenRouterClient` остаётся как тонкая обёртка для back-compat.

Поведение:
    - `chat(messages, models, ...)` — пробует список моделей по очереди.
      На 429/502/503 / timeout-ах / connection-error'ах переходит к
      следующей; на других HTTP-ошибках/ValueError'ах сразу возвращает
      ошибку без попыток оставшихся моделей.
    - `probe(model)` — короткий запрос «Ответь одним словом: OK», для
      `/health` и `/test_api`.

Конструктор `OpenAICompatibleClient`:
    api_key:       Bearer-токен провайдера.
    base_url:      URL без хвоста (например https://api.deepseek.com/v1).
                   К нему будет добавлено `/chat/completions`.
    extra_headers: дополнительные заголовки (например, HTTP-Referer
                   и X-Title для OpenRouter).
    provider_name: имя провайдера для логов (например, 'openrouter').
    timeout:       таймаут одного HTTP-запроса (сек).
    transport:     опциональный httpx.AsyncBaseTransport для unit-тестов.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx


logger = logging.getLogger(__name__)


# Перехватываются `chat()` и `probe()` для перехода к следующей модели.
_RETRY_STATUSES = {429, 502, 503, 504}


class OpenAIError(Exception):
    """Ошибка OpenAI-compatible API: либо все модели отвалились, либо
    явный HTTP-error (4xx, не 429), либо отсутствует ключ."""


# Back-compat alias — все существующие импорты `OpenRouterError`
# продолжают работать. Новый код должен использовать OpenAIError.
OpenRouterError = OpenAIError


class OpenAICompatibleClient:
    """Async-клиент для любого OpenAI-compatible эндпоинта."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        extra_headers: Optional[dict[str, str]] = None,
        provider_name: str = "openai-compatible",
        timeout: float = 90.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        # base_url без хвостового слэша — `_chat_completions_url`
        # добавит `/chat/completions` сам.
        self.base_url = (base_url or "").rstrip("/")
        self.extra_headers = dict(extra_headers or {})
        self.provider_name = provider_name or "openai-compatible"
        self.timeout = timeout
        self._transport = transport

    # ─── Public helpers ────────────────────────────────────────────

    @property
    def key_present(self) -> bool:
        """True, если ключ заполнен и не выглядит как placeholder."""
        if not self.api_key:
            return False
        lower = self.api_key.lower()
        return "ваш_ключ" not in lower and "your_key" not in lower

    @property
    def key_redacted(self) -> str:
        """Маскированный ключ для логов и /test_api."""
        if not self.api_key:
            return "(empty)"
        if len(self.api_key) <= 16:
            return self.api_key[:4] + "..."
        return f"{self.api_key[:12]}...{self.api_key[-4:]}"

    # ─── Core call ─────────────────────────────────────────────────

    @property
    def _chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)
        return headers

    async def _post_chat(
        self,
        messages: list,
        model: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Один HTTP-запрос на /chat/completions. Не делает retry."""
        async with httpx.AsyncClient(
            timeout=self.timeout, transport=self._transport,
        ) as client:
            r = await client.post(
                self._chat_completions_url,
                headers=self._headers(),
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            r.raise_for_status()
            data = r.json()
            # OpenRouter иногда возвращает HTTP 200 + error в теле (рейтлимит
            # провайдера, недоступный маршрут, content-policy). Это soft-
            # fail — верхний уровень должен уйти на следующую модель.
            if isinstance(data, dict) and data.get("error"):
                err = data["error"]
                msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                raise OpenAIError(
                    f"{self.provider_name}/{model}: 200 OK + error в теле: {str(msg)[:200]}"
                )
            try:
                choices = data["choices"]
            except (KeyError, TypeError) as exc:
                raise OpenAIError(
                    f"{self.provider_name}/{model}: ответ без поля 'choices'"
                ) from exc
            if not choices:
                raise OpenAIError(f"{self.provider_name}/{model}: пустой 'choices'")
            first = choices[0] if isinstance(choices[0], dict) else None
            message = first.get("message") if first else None
            if not isinstance(message, dict):
                raise OpenAIError(f"{self.provider_name}/{model}: нет 'message' в choices[0]")
            content = message.get("content")
            # Авто-роутеры (например openrouter/free) иногда отвечают
            # content=None (выбрали модель под rate-limit). Без этой
            # проверки это приводило к AttributeError на .strip().
            if content is None:
                raise OpenAIError(
                    f"{self.provider_name}/{model}: content=None (модель ничего "
                    "не сгенерировала, вероятно rate-limit на стороне провайдера)"
                )
            if not isinstance(content, str):
                raise OpenAIError(
                    f"{self.provider_name}/{model}: content неожиданного типа "
                    f"{type(content).__name__}"
                )
            return content.strip()

    async def chat(
        self,
        messages: list,
        models: list[str],
        *,
        temperature: float = 0.1,
        max_tokens: int = 3000,
    ) -> tuple[str, str]:
        """Делает запрос с fallback'ом по списку моделей.

        Возвращает `(content, used_model)`. Если все модели отвалились с
        retry-статусами (429/502/503/504) — поднимает `OpenRouterError`
        с описанием последней ошибки. Если первая попытка вернула
        не-retry HTTP-ошибку (например, 400 / 401 / 403) — поднимает
        сразу, без перебора остальных моделей.
        """
        if not self.key_present:
            raise OpenAIError(f"Ключ провайдера {self.provider_name} не задан")
        if not models:
            raise OpenAIError("Список моделей пуст")

        last_err: str = ""
        statuses: list[Optional[int]] = []
        for model in models:
            try:
                content = await self._post_chat(
                    messages, model,
                    temperature=temperature, max_tokens=max_tokens,
                )
                logger.info("%s: %s ответил (%d символов)", self.provider_name, model, len(content))
                return content, model
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                body = e.response.text[:200]
                last_err = f"[{self.provider_name}/{model}] HTTP {status}: {body}"
                if status in _RETRY_STATUSES:
                    logger.info(
                        "%s: %s вернул HTTP %d, пробую следующую модель",
                        self.provider_name, model, status,
                    )
                    statuses.append(status)
                    continue
                # 4xx (auth, неверный модель-id) — нет смысла перебирать дальше
                logger.warning("%s: %s вернул HTTP %d (не retry-able)", self.provider_name, model, status)
                raise OpenAIError(last_err) from e
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_err = f"[{self.provider_name}/{model}] {type(e).__name__}: {str(e)[:160]}"
                logger.info("%s: %s timeout/network, пробую следующую модель", self.provider_name, model)
                statuses.append(None)
                continue
            except OpenAIError as e:
                # неверный формат / content=None / 200 OK + error в теле —
                # soft fail, пробуем следующую модель.
                last_err = f"[{self.provider_name}/{model}] {e}"
                logger.info("%s: %s soft-fail, пробую следующую: %s", self.provider_name, model, str(e)[:120])
                statuses.append(None)
                continue
            except Exception as e:  # noqa: BLE001
                last_err = f"[{self.provider_name}/{model}] {type(e).__name__}: {str(e)[:160]}"
                logger.exception("%s: непредвиденная ошибка на %s", self.provider_name, model)
                statuses.append(None)
                continue
        # Если все модели вернули 429 — это исчерпание квоты.
        if statuses and all(s == 429 for s in statuses):
            raise OpenAIError(
                f"Все модели провайдера {self.provider_name} вернули HTTP 429 "
                "(исчерпана дневная квота). Варианты: (1) пополнить баланс "
                "провайдера, (2) подождать 24 часа, (3) в server/cloud/.env "
                "добавить второго провайдера (CLOUD_PROVIDERS=openrouter,deepseek,...) "
                "или платные модели в список."
            )
        raise OpenAIError(f"Все модели {self.provider_name} недоступны. Последняя ошибка: {last_err}")

    async def probe(self, model: str) -> str:
        """Минимальный probe-запрос для /health и /test_api."""
        content, _ = await self.chat(
            [{"role": "user", "content": "Ответь одним словом: OK"}],
            [model],
            temperature=0.0,
            max_tokens=32,
        )
        return content


class OpenRouterClient(OpenAICompatibleClient):
    """Back-compat: тонкая обёртка над OpenAICompatibleClient с
    дефолтами для OpenRouter (base_url + HTTP-Referer + X-Title).

    Сохранена для существующего кода и тестов. Новый код должен
    использовать `OpenAICompatibleClient` напрямую или `ProviderConfig`
    из `shared.providers`.
    """

    def __init__(
        self,
        api_key: str,
        *,
        referer: str = "http://localhost",
        title: str = "AI LibreOffice Suggester",
        timeout: float = 90.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        super().__init__(
            api_key,
            base_url="https://openrouter.ai/api/v1",
            extra_headers={"HTTP-Referer": referer, "X-Title": title},
            provider_name="openrouter",
            timeout=timeout,
            transport=transport,
        )
        # Сохраняем атрибуты для back-compat — тесты могут читать
        # client.referer / client.title.
        self.referer = referer
        self.title = title


# ─── Preset selection (cloud-side analog of LLM_PRESET) ───────────────


CLOUD_PRESETS: dict[str, dict[str, str | list[str]]] = {
    "A": {
        "PRIMARY": "openrouter/free",
        "DESCRIPTION": (
            "openrouter/free auto-router — автоматически выбирает живую free-модель. "
            "Самый надёжный preset: при недоступности одной модели OpenRouter "
            "сам переключит на другую."
        ),
        "FALLBACK": [
            "qwen/qwen3-next-80b-a3b-instruct:free",
            "google/gemma-4-31b-it:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "openai/gpt-oss-120b:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
        ],
    },
    "B": {
        "PRIMARY": "qwen/qwen3-next-80b-a3b-instruct:free",
        "DESCRIPTION": (
            "Qwen3-Next 80B A3B MoE instruct — multilingual, явно поддерживает "
            "русский. Контекст 262K. Хороший выбор для русскоязычных официальных "
            "документов."
        ),
        "FALLBACK": [
            "openrouter/free",
            "google/gemma-4-31b-it:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "openai/gpt-oss-120b:free",
        ],
    },
    "C": {
        "PRIMARY": "google/gemma-4-31b-it:free",
        "DESCRIPTION": (
            "Gemma 4 31B Instruct от Google DeepMind — dense 30.7B, multilingual, "
            "контекст 256K. Сильный по русскому, configurable thinking-mode "
            "(отключаем для GEC)."
        ),
        "FALLBACK": [
            "openrouter/free",
            "qwen/qwen3-next-80b-a3b-instruct:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
        ],
    },
    "D": {
        "PRIMARY": "nvidia/nemotron-3-super-120b-a12b:free",
        "DESCRIPTION": (
            "NVIDIA Nemotron 3 Super — 120B MoE, 12B активных параметров, "
            "контекст 262K. Hybrid Mamba-Transformer-MoE, сильный reasoning. "
            "Менее заточен под русский, но хороший для сложного контекста."
        ),
        "FALLBACK": [
            "openrouter/free",
            "qwen/qwen3-next-80b-a3b-instruct:free",
            "google/gemma-4-31b-it:free",
            "openai/gpt-oss-120b:free",
        ],
    },
}


def resolve_models(preset: str, override_models: Optional[list[str]] = None) -> list[str]:
    """Возвращает список моделей для перебора `chat()`.

    Если `override_models` непустой — используем его (явный список из env).
    Иначе берём preset (A/B/C/D); неизвестный preset → A.
    Первая модель = PRIMARY (preset.PRIMARY), затем FALLBACK.
    """
    if override_models:
        return list(override_models)
    cfg = CLOUD_PRESETS.get(preset.upper(), CLOUD_PRESETS["A"])
    models: list[str] = [cfg["PRIMARY"]]  # type: ignore[list-item]
    for m in cfg.get("FALLBACK", []):  # type: ignore[union-attr]
        if m not in models:
            models.append(m)
    return models
