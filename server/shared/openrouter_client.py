"""HTTP-клиент OpenRouter с fallback-цепочкой моделей.

Извлечён из `server/cloud/main.py` (v2.1) — раньше логика fallback'а жила
прямо в `/suggest` и `/health`. Теперь это переиспользуемый клиент с
chat-завершением, probe-методом и явным контролем retry-семантики.

Поведение:
    - `chat(messages, models, ...)` — пробует список моделей по очереди.
      На 429/502/503 / timeout-ах / connection-error'ах переходит к
      следующей; на других HTTP-ошибках/ValueError'ах сразу возвращает
      ошибку без попыток оставшихся моделей.
    - `probe(model)` — короткий запрос «Ответь одним словом: OK», для
      `/health` и `/test_api`.

Конструктор:
    api_key:   ключ OpenRouter (sk-or-v1-...)
    referer:   `HTTP-Referer` header (для квот OpenRouter)
    title:     `X-Title` header
    timeout:   таймаут одного HTTP-запроса (сек)
    transport: опциональный httpx.AsyncBaseTransport для unit-тестов
               (мок без сети). По умолчанию — None.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx


logger = logging.getLogger(__name__)


# Перехватываются `chat()` и `probe()` для перехода к следующей модели.
_RETRY_STATUSES = {429, 502, 503, 504}


class OpenRouterError(Exception):
    """Ошибка OpenRouter API: либо все модели отвалились, либо явный
    HTTP-error (4xx, не 429), либо отсутствует ключ."""


class OpenRouterClient:
    """Async-клиент OpenRouter с fallback-цепочкой моделей."""

    def __init__(
        self,
        api_key: str,
        *,
        referer: str = "http://localhost",
        title: str = "AI LibreOffice Suggester",
        timeout: float = 90.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.referer = referer
        self.title = title
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

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.referer,
            "X-Title": self.title,
        }

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
                "https://openrouter.ai/api/v1/chat/completions",
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
                raise OpenRouterError(
                    f"{model}: 200 OK + error в теле: {str(msg)[:200]}"
                )
            try:
                choices = data["choices"]
            except (KeyError, TypeError) as exc:
                raise OpenRouterError(
                    f"{model}: ответ без поля 'choices'"
                ) from exc
            if not choices:
                raise OpenRouterError(f"{model}: пустой 'choices'")
            first = choices[0] if isinstance(choices[0], dict) else None
            message = first.get("message") if first else None
            if not isinstance(message, dict):
                raise OpenRouterError(f"{model}: нет 'message' в choices[0]")
            content = message.get("content")
            # Ключевой фикс: openrouter/free auto-router иногда отвечает
            # content=None (например, выбрал модель под rate-limit). До
            # фикса это приводило к AttributeError в _chat_with_fallback.
            if content is None:
                raise OpenRouterError(
                    f"{model}: content=None (модель ничего не сгенерировала, "
                    "вероятно rate-limit на стороне провайдера)"
                )
            if not isinstance(content, str):
                raise OpenRouterError(
                    f"{model}: content неожиданного типа "
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
            raise OpenRouterError("OPENROUTER_API_KEY не задан")
        if not models:
            raise OpenRouterError("Список моделей пуст")

        last_err: str = ""
        statuses: list[Optional[int]] = []
        for model in models:
            try:
                content = await self._post_chat(
                    messages, model,
                    temperature=temperature, max_tokens=max_tokens,
                )
                logger.info("OpenRouter: %s ответил (%d символов)", model, len(content))
                return content, model
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                body = e.response.text[:200]
                last_err = f"[{model}] HTTP {status}: {body}"
                if status in _RETRY_STATUSES:
                    logger.info(
                        "OpenRouter: %s вернул HTTP %d, пробую следующую модель",
                        model, status,
                    )
                    statuses.append(status)
                    continue
                # 4xx (auth, неверный модель-id) — нет смысла перебирать дальше
                logger.warning("OpenRouter: %s вернул HTTP %d (не retry-able)", model, status)
                raise OpenRouterError(last_err) from e
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_err = f"[{model}] {type(e).__name__}: {str(e)[:160]}"
                logger.info("OpenRouter: %s timeout/network, пробую следующую модель", model)
                statuses.append(None)
                continue
            except OpenRouterError as e:
                # неверный формат / content=None / 200 OK + error в теле —
                # soft fail, пробуем следующую модель.
                last_err = f"[{model}] {e}"
                logger.info("OpenRouter: %s soft-fail, пробую следующую: %s", model, str(e)[:120])
                statuses.append(None)
                continue
            except Exception as e:  # noqa: BLE001
                last_err = f"[{model}] {type(e).__name__}: {str(e)[:160]}"
                logger.exception("OpenRouter: непредвиденная ошибка на %s", model)
                statuses.append(None)
                continue
        # Если все модели вернули 429 — это исчерпание квоты free-tier.
        if statuses and all(s == 429 for s in statuses):
            raise OpenRouterError(
                "Все модели OpenRouter вернули HTTP 429 (исчерпана дневная "
                "квота free-tier). Варианты: (1) пополнить баланс OpenRouter "
                "на $10 — это открывает 1000 запросов/день по всем :free моделям, "
                "(2) подождать 24 часа (квота обновляется), (3) в server/cloud/.env "
                "прописать OPENROUTER_MODELS=... с платной моделью (например "
                "meta-llama/llama-3.3-70b-instruct без суффикса :free)."
            )
        raise OpenRouterError(f"Все модели недоступны. Последняя ошибка: {last_err}")

    async def probe(self, model: str) -> str:
        """Минимальный probe-запрос для /health и /test_api."""
        content, _ = await self.chat(
            [{"role": "user", "content": "Ответь одним словом: OK"}],
            [model],
            temperature=0.0,
            max_tokens=32,
        )
        return content


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
