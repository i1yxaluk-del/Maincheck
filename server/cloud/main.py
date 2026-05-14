"""
Облачный сервер AI LibreOffice Suggester (OpenRouter) — v2.2.

Назначение: использовать сильную сетевую LLM (Qwen3-Next 80B, Gemma 4 31B,
Nemotron 3 Super 120B, openrouter/free auto-router) как полностью
автономный корректор русских официальных документов. Сервер сознательно
устроен проще локального — сетевая модель сама лучше справляется с
большинством классов ошибок, поэтому local-specific костыли
(морф-фильтр, морф-детектор, sage, LanguageTool, few-shot retrieval)
здесь намеренно отсутствуют.

Архитектура (v2.2, в порядке pipeline):
    1. Нормализация переносов строк (Shift+Enter / разрыв абзаца) — см.
       раздел «Shift+Enter»: одиночный \\n — мягкий перенос внутри
       абзаца, двойной \\n\\n — разрыв абзаца. Эта же конвенция
       сохраняется в исправленном тексте.
    2. RAG-контекст (RAG_ENABLED=true) — фрагменты НПА РФ из
       `data/rag_store/`, проиндексированные ранее через rag_cli.
       Подсказка модели — не цитировать в CHANGES напрямую.
    3. Пользовательский словарь (USER_DICT_ENABLED=true) —
       whitelist терминов в system-prompt + REST API `/dict/*`.
       Модель не должна «исправлять» термины из этого списка.
    4. OpenRouter chat с fallback по CLOUD_PRESET (A/B/C/D).
    5. Безопасный пост-процессинг (модель уже хорошая, минимизируем):
        • strip-thinking — обрезка `<think>…</think>` блоков;
        • безопасные защитные фильтры формата (формат ответа,
          идемпотентные правки X→X, drop-not-in-text, renumber);
        • восстановление структуры переносов (см. п.1).

Backward compatibility:
    • Без новых env-флагов поведение совпадает с v1.6.0 (только список
      моделей актуализирован под free-tier OpenRouter мая 2026).
    • Старые тесты `test_cloud_suggest_with_mocked_openrouter`,
      `test_cloud_metrics`, `test_cloud_missing_key` продолжают работать.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from shared.audit import AuditStore, Timer, count_changes  # noqa: E402
from shared.logging_setup import setup_logger  # noqa: E402
from shared.openrouter_client import (  # noqa: E402
    CLOUD_PRESETS,
    OpenAICompatibleClient,
    OpenAIError,
    OpenRouterClient,
    OpenRouterError,
)
from shared.providers import (  # noqa: E402
    ProviderConfig,
    load_providers_from_env,
)
from shared.postprocess import (  # noqa: E402
    _drop_changes_not_in_text,
    _drop_eyo_substitutions,
    _drop_idempotent_changes,
    _renumber_changes,
    _strip_thinking,
    _undo_eyo_in_corrected_block,
)


load_dotenv()


# ─── Multi-provider конфиг (v2.2.3+) ──────────────────────────────────
#
# `CLOUD_PROVIDERS=openrouter,deepseek,fireworks` в .env — список
# OpenAI-compatible провайдеров для перебора. Для каждого — отдельные
# env-переменные <NAME>_API_KEY, <NAME>_BASE_URL (опц.), <NAME>_MODELS.
# Если CLOUD_PROVIDERS не задан — дефолт `openrouter` (back-compat).

OPENROUTER_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT", "90"))
OPENROUTER_TEMPERATURE = float(os.getenv("OPENROUTER_TEMPERATURE", "0.1"))
OPENROUTER_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "3000"))

# CLOUD_PRESET (A/B/C/D) — fallback для openrouter, если не задан OPENROUTER_MODELS.
CLOUD_PRESET = os.getenv("CLOUD_PRESET", "A").strip().upper()
if CLOUD_PRESET not in CLOUD_PRESETS:
    CLOUD_PRESET = "A"

_RAW_PROVIDERS: list[ProviderConfig] = load_providers_from_env()

# Только провайдеры с заполненным ключом и непустым списком моделей.
PROVIDERS: list[ProviderConfig] = [
    p for p in _RAW_PROVIDERS if p.key_present and p.models
]

# Плоский список моделей в порядке (provider1.models, provider2.models, ...).
# Используется в `_chat_with_fallback` для последовательного перебора.
MODELS: list[str] = [m for p in PROVIDERS for m in p.models]

# Роутинг: model_id → (provider, client). Первая регистрация выигрывает,
# поэтому порядок CLOUD_PROVIDERS определяет какой провайдер выберется,
# если модель с одним именем настроена в нескольких провайдерах.
_PROVIDER_CLIENTS: dict[str, OpenAICompatibleClient] = {}
_MODEL_ROUTING: dict[str, tuple[ProviderConfig, OpenAICompatibleClient]] = {}
for _p in PROVIDERS:
    _client = _p.build_client(timeout=OPENROUTER_TIMEOUT)
    _PROVIDER_CLIENTS[_p.name] = _client
    for _m in _p.models:
        if _m not in _MODEL_ROUTING:
            _MODEL_ROUTING[_m] = (_p, _client)

# Back-compat: первый openrouter-клиент (или просто первый, если openrouter
# не настроен). Используется только для `/test_api`, `_key_missing` и старыми
# тестами через `cloud_module.or_client`.
_first_or = next((c for n, c in _PROVIDER_CLIENTS.items() if n.lower() == "openrouter"), None)
or_client = _first_or or next(iter(_PROVIDER_CLIENTS.values()), None) or OpenRouterClient(
    api_key="",
)


# ─── RAG (опционально, RAG_ENABLED=true) ──────────────────────────────

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
RAG_ENABLED = os.getenv("RAG_ENABLED", "false").lower() in ("1", "true", "yes", "on")
RAG_STORE_DIR = os.getenv("RAG_STORE_DIR", "data/rag_store")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
RAG_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")


# ─── User dict (опционально, USER_DICT_ENABLED=true) ─────────────────

USER_DICT_ENABLED = os.getenv("USER_DICT_ENABLED", "true").lower() in ("1", "true", "yes", "on")


# ─── Логи и аудит ─────────────────────────────────────────────────────
# Уровень и retention настраиваются через env-переменные внутри
# setup_logger() / AuditStore() — здесь только инициализация.

logger = setup_logger("ai_suggester.cloud")
audit = AuditStore()

logger.info(
    "Cloud server v2.2.3 starting: providers=%s total_models=%d",
    [p.name for p in PROVIDERS] or "(none configured)", len(MODELS),
)
for _p in _RAW_PROVIDERS:
    if not _p.key_present:
        logger.warning("Provider %s: ключ не задан — провайдер пропущен", _p.name)
        continue
    if not _p.models:
        logger.warning("Provider %s: список моделей пуст — провайдер пропущен", _p.name)
        continue
    _c = _PROVIDER_CLIENTS[_p.name]
    logger.info(
        "Provider %s: key=%s, base_url=%s, models=%d (primary=%s)",
        _p.name, _c.key_redacted, _p.base_url, len(_p.models), _p.models[0],
    )


# ─── Lazy singletons ──────────────────────────────────────────────────

_rag_store = None
_rag_embedder = None
_user_dict = None

if RAG_ENABLED:
    try:
        from shared.rag_store import OllamaEmbedder, RagStore  # noqa: E402

        _rag_store = RagStore(RAG_STORE_DIR)
        _rag_embedder = OllamaEmbedder(model=RAG_EMBED_MODEL, base_url=OLLAMA_URL)
        logger.info(
            "RAG включён: store=%s, embedder=%s, docs=%d",
            RAG_STORE_DIR, RAG_EMBED_MODEL, len(_rag_store.docs),
        )
    except Exception as e:
        logger.warning("RAG не удалось инициализировать: %s", e)
        _rag_store = None

if USER_DICT_ENABLED:
    try:
        from shared.user_dict import get_user_dict  # noqa: E402

        _user_dict = get_user_dict()
        logger.info(
            "UserDict включён: загружено %d слов(а) из %s",
            len(_user_dict.list_words()), _user_dict.path,
        )
    except Exception as e:
        logger.warning("UserDict не удалось инициализировать: %s", e)
        _user_dict = None


# ─── SYSTEM_PROMPT (v2.2 — сильный сетевой корректор) ────────────────

SYSTEM_PROMPT = """Ты — корректор русского языка для официальных документов государственных органов РФ. Твоя задача — найти и исправить ВСЕ ошибки в текстах служебной переписки, организационно-распорядительных и правовых документах.

ОБЯЗАТЕЛЬНО ИСПРАВЛЯЙ ОШИБКИ ВСЕХ ВИДОВ:

1. ОРФОГРАФИЯ — опечатки, удвоение/пропуск букв, слитное/раздельное/дефисное написание, перепутанные буквы (а/о, и/е), приставки (пре-/при-, не-/ни-), окончания.

2. ПУНКТУАЦИЯ — запятые при однородных членах, обособленных оборотах (причастный, деепричастный, уточняющий), вводных словах, придаточных; тире (длинное — а не дефис -); точка с запятой между сложными однородными; кавычки (елочки «», лапки ""); постановка точек в конце нумерованных пунктов; пробелы вокруг знаков.

3. ГРАММАТИКА И СОГЛАСОВАНИЕ:
   • падежные формы — «согласно приказу» (дат.), «вопреки решению» (дат.), «по приезде» (предл.), «по окончании», «по истечении»;
   • согласование однородных членов с главным словом в роде, числе и падеже, даже если оно стоит за несколько слов («актов, …, не предусмотренных», а не «не предусмотренной»);
   • числительные с существительными — «во 2-м квартале» (ед., предл.), а не «во 2-м кварталах»;
   • прилагательное с существительным в роде/числе/падеже;
   • временные формы и залог глаголов в сложных конструкциях;
   • неполные предложения, согласование местоимений с антецедентом.

4. СТИЛЬ И ЛОГИКА:
   • канцелярит уместен, но избегай тавтологии («осуществляет осуществление»), плеоназмов («первый по приоритету первый»), бюрократических оборотов-паразитов;
   • согласование времен в сложных предложениях;
   • устранение двусмысленности и противоречий между смежными абзацами;
   • точные числовые ссылки (даты, номера приказов, реквизиты НПА — если в тексте уже упомянуты);
   • единообразие сокращений и аббревиатур внутри одного текста.

5. ОФИЦИАЛЬНО-ДЕЛОВАЯ ПЕРЕПИСКА РФ — по ГОСТ Р 7.0.97-2016 (Система стандартов по информации, библиотечному и издательскому делу):
   • реквизиты — «исх. №», «вх. №», «п/п»;
   • устойчивые формулы — «прошу Вас», «направляю Вам», «в соответствии с», «во исполнение»;
   • правильная иерархия — «Министерство → Управление → Отдел»;
   • НПА оформляются как «Федеральный закон от 06.03.2006 № 35-ФЗ "О противодействии терроризму"», постановления Правительства РФ — с датой и номером.

ЧТО ЗАПРЕЩЕНО ТРОГАТЬ:
• аббревиатуры и сокращения, имена собственные, наименования организаций и должностей;
• ведомственные термины, профессиональную лексику, торговые марки;
• авторскую стилистику, если она не нарушает норму;
• БУКВА Ё СТРОГО ЗАПРЕЩЕНА К ПРАВКаМ. Е/ё в русском тексте взаимозаменяемы и ни одно из этих написаний не является ошибкой. ПРОСЫБА: НИКОГДА не предлагай замены, отличающиеся ТОЛЬКО наличием/отсутствием буквы ё (например «ведёние» → «ведение» или наоборот). При сомнениях — оставь как в оригинале.
• общий смысл, состав сведений, фактическую информацию (даты, номера, имена) не менять;
• если фрагмент уже грамматически корректен — оставь как есть, даже если предпочёл бы переформулировать.

КОНТЕКСТ:
• первое сообщение — «Контекст» (текст ДО проверяемого фрагмента, только для понимания стиля и связности; в нём ничего не исправляй).
• «ТЕКСТ ДЛЯ ПРОВЕРКИ» — фрагмент, в котором ищешь ошибки. Может быть от 1 предложения до нескольких абзацев. Все ошибки во всём фрагменте — единым проходом.

СТРУКТУРА ПЕРЕНОСОВ СТРОК (ВАЖНО ДЛЯ КЛИЕНТА):
• одиночный \\n внутри абзаца — это мягкий перенос строки (Shift+Enter), оставляй на тех же местах;
• двойной \\n\\n — граница абзаца, оставляй на тех же местах;
• НЕ объединяй абзацы в один, НЕ разрывай абзац на несколько.

ФОРМАТ ОТВЕТА (СТРОГО, без какого-либо текста до или после, без рассуждений):
===CORRECTED===
<полный исправленный текст с сохранёнными переносами строк и абзацев>
===CHANGES===
1. «было» → «стало» | краткая причина (5–15 слов, по-русски)
2. ...
===END===

ПРАВИЛА ЦИТИРОВАНИЯ В CHANGES (клиент применяет правки ПОИСКОМ по тексту):
• «было» — ТОЧНО, побуквенно, как в исходном тексте (с учётом регистра);
• БЕЗ многоточия, БЕЗ сокращений, БЕЗ перефразирования;
• «стало» — фрагмент той же длины с применённой правкой;
• если правка касается одного знака препинания — включи в «было»/«стало» одно-два слова слева и справа от знака, чтобы клиент мог найти контекст;
• если в тексте есть НПА — комментарий может ссылаться на норму («п. 3 ст. 12 Закона № 35-ФЗ»), но саму ссылку НЕ выдумывай — только если такая норма явно упомянута в RAG-контексте или в самом тексте.

Если ошибок нет:
===CORRECTED===
<исходный текст без изменений>
===CHANGES===
1. Ошибок не найдено. Текст соответствует нормам.
===END==="""


app = FastAPI(title="AI LibreOffice Suggester — Cloud", version="2.2.0")


# ─── Shift+Enter / структура переносов строк ──────────────────────────


def _normalize_line_breaks(text: str) -> str:
    """Приводит входной текст к канонической форме переносов:

    LibreOffice getString() возвращает paragraph-break как \\r или \\r\\n
    (платформо-зависимо), а Shift+Enter (soft return) — как \\n или
    U+2028. Конвенция, общая с client-side ApplyWholeReplace:

    • двойной \\n\\n  — граница абзаца;
    • одиночный \\n   — мягкий перенос внутри абзаца.

    Превращения:
    • \\r\\n → \\n\\n (paragraph break, Windows);
    • \\r    → \\n\\n (paragraph break, macOS classic / LO Linux);
    • U+2028 → \\n     (Unicode line separator → soft return);
    • уже-\\n не трогаем; 3+ подряд схлопываем до \\n\\n.

    До v2.2 одиночный Chr(10) от Shift+Enter после round-trip через LLM
    мог склеиваться или, наоборот, превращаться в paragraph break при
    ApplyWholeReplace в Main.xba — пользователь видел «разъединение»
    одного фрагмента на несколько абзацев. См. ЖУРНАЛ_v1.6.md v2.2.
    """
    if not text:
        return text
    text = text.replace("\r\n", "\n\n")
    text = text.replace("\r", "\n\n")
    text = text.replace("\u2028", "\n")
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text


# ─── Backward-compatible helpers (тест test_cloud_suggest_with_mocked_openrouter
# подменяет cloud_module.call_model) ──────────────────────────────────


def _key_missing() -> bool:
    """True, если ни один провайдер не настроен (нет ни одного валидного ключа)."""
    return not PROVIDERS


def _client_for_model(model: str) -> tuple[ProviderConfig, OpenAICompatibleClient]:
    """Находит провайдера и клиент для конкретной модели.

    Если модель не зарегистрирована ни в одном провайдере — фоллбек на
    `or_client` (back-compat: тесты, мокирующие `call_model`, могут
    подсунуть произвольную модель).
    """
    entry = _MODEL_ROUTING.get(model)
    if entry:
        return entry
    # Back-compat: модель не из конфигурации (например, тестовая) —
    # используем openrouter-клиент по умолчанию.
    fallback = PROVIDERS[0] if PROVIDERS else ProviderConfig(
        name="openrouter", base_url="https://openrouter.ai/api/v1",
        api_key="", models=[],
    )
    return fallback, or_client


async def call_model(messages: list, model: str) -> str:
    """Один HTTP-запрос на одну модель (backward-compat для тестов).

    Бросает `httpx.HTTPStatusError` на 4xx/5xx, чтобы верхний уровень мог
    решить, переходить ли к следующей модели по fallback-цепочке. Тесты
    подменяют эту функцию для мокирования.

    Роутинг провайдера — по `_MODEL_ROUTING`, заполненному из
    `CLOUD_PROVIDERS`. Если модель не привязана к провайдеру, используется
    `or_client` (back-compat).
    """
    _, client = _client_for_model(model)
    return await client._post_chat(
        messages, model,
        temperature=OPENROUTER_TEMPERATURE,
        max_tokens=OPENROUTER_MAX_TOKENS,
    )


async def _chat_with_fallback(messages: list) -> tuple[str, str]:
    """Перебирает MODELS до первого успешного ответа. Возвращает
    (content, used_model). На полном фейле — поднимает OpenRouterError.

    Multi-provider: MODELS — плоский список из всех настроенных
    провайдеров. Каждая модель роутится через `_MODEL_ROUTING` в
    соответствующий клиент. Использует `call_model` для каждой попытки —
    это позволяет тестам мокировать одну точку входа.
    """
    last_err = "нет ответа"
    statuses: list[int | None] = []
    for model in MODELS:
        provider, _ = _client_for_model(model)
        prefix = f"{provider.name}/{model}"
        try:
            content = await call_model(messages, model)
            return content, model
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            last_err = f"[{prefix}] HTTP {status}: {e.response.text[:160]}"
            if status in (429, 502, 503, 504):
                logger.info("%s: HTTP %d, пробую следующую", prefix, status)
                statuses.append(status)
                continue
            logger.warning("%s: HTTP %d (не retry)", prefix, status)
            raise OpenRouterError(last_err) from e
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            last_err = f"[{prefix}] {type(e).__name__}: {str(e)[:120]}"
            logger.info("%s: timeout/network, пробую следующую", prefix)
            statuses.append(None)
            continue
        except OpenRouterError as e:
            # 200 OK + content=None / error в теле / неверный формат —
            # soft-fail, без traceback в лог.
            last_err = f"[{prefix}] {e}"
            logger.info("%s: soft-fail: %s", prefix, str(e)[:160])
            statuses.append(None)
            continue
        except Exception as e:  # noqa: BLE001
            last_err = f"[{prefix}] {type(e).__name__}: {str(e)[:160]}"
            logger.exception("%s: непредвиденная ошибка", prefix)
            statuses.append(None)
            continue
    if statuses and all(s == 429 for s in statuses):
        provider_names = ", ".join(p.name for p in PROVIDERS) or "openrouter"
        raise OpenRouterError(
            f"Все модели всех провайдеров ({provider_names}) вернули HTTP 429 "
            "(исчерпана дневная квота на всех настроенных провайдерах). Варианты: "
            "(1) пополнить баланс на одном из провайдеров (для OpenRouter — $10 "
            "даёт 1000 запросов/день по всем :free моделям), "
            "(2) подождать 24 часа (квоты обновляются), "
            "(3) в server/cloud/.env добавить ещё одного провайдера в "
            "CLOUD_PROVIDERS=openrouter,deepseek,fireworks,... и указать для него "
            "<NAME>_API_KEY + <NAME>_MODELS — fallback автоматически переключится "
            "на резервного провайдера."
        )
    raise OpenRouterError(f"Все модели недоступны. Последняя ошибка: {last_err}")


# ─── /rag_context ─────────────────────────────────────────────────────


def _rag_context(text: str) -> str:
    """Возвращает дополнительный блок с фрагментами из RAG-хранилища.

    No-op, если RAG_ENABLED=false или хранилище пустое.
    """
    if not (_rag_store and _rag_embedder):
        return ""
    try:
        hits = _rag_store.search(text, top_k=RAG_TOP_K, embedder=_rag_embedder)
    except Exception as e:
        logger.warning("RAG поиск провалился: %s", e)
        return ""
    if not hits:
        return ""
    parts = [
        "ПРИМЕНИМЫЕ НОРМАТИВНЫЕ ФРАГМЕНТЫ "
        "(используй как справку и для ссылок на НПА в CHANGES; НЕ цитируй дословно):",
    ]
    for h in hits:
        parts.append(f"— [{h['doc_id']}] {h['text'][:600]}")
    return "\n".join(parts)


# ─── Endpoints ────────────────────────────────────────────────────────


@app.get("/health", response_class=PlainTextResponse)
async def health():
    if _key_missing():
        return (
            "ОШИБКА: ни один провайдер не настроен. Задайте хотя бы "
            "OPENROUTER_API_KEY + OPENROUTER_MODELS в server/cloud/.env, "
            "либо CLOUD_PROVIDERS=... с другими провайдерами."
        )
    probe_msgs = [{"role": "user", "content": "Ответь одним словом: OK"}]
    last_err = "нет ответа"
    for model in MODELS:
        provider, _ = _client_for_model(model)
        prefix = f"{provider.name}/{model}"
        try:
            ans = await call_model(probe_msgs, model)
            extras = []
            if RAG_ENABLED and _rag_store:
                extras.append(f"RAG: {len(_rag_store.docs)} документов")
            if _user_dict is not None:
                extras.append(f"Dict: {len(_user_dict.list_words())} слов")
            suffix = (" | " + " | ".join(extras)) if extras else ""
            return f"OK | Работает: {prefix} | Ответ: {ans[:40]}{suffix}"
        except httpx.HTTPStatusError as e:
            last_err = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            if e.response.status_code in (429, 502, 503, 504):
                continue
            return f"ОШИБКА HTTP на {prefix}: {last_err}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
            continue
    return f"ОШИБКА: Все модели недоступны. Последняя ошибка: {last_err}"


@app.get("/test_api", response_class=PlainTextResponse)
async def test_api():
    if _key_missing():
        return (
            "ОШИБКА: ни один провайдер не настроен. Задайте хотя бы "
            "OPENROUTER_API_KEY + OPENROUTER_MODELS в server/cloud/.env."
        )
    lines: list[str] = []
    for provider in PROVIDERS:
        client = _PROVIDER_CLIENTS[provider.name]
        header = f"Provider: {provider.name} | key={client.key_redacted} | base_url={provider.base_url}"
        if provider.name.lower() == "openrouter":
            header += f" | preset={CLOUD_PRESET}"
        lines.append(header)
        for model in provider.models:
            try:
                ans = await call_model(
                    [{"role": "user", "content": "Ответь одним словом: OK"}], model,
                )
                lines.append(f"  [OK]   {model}\n         → {ans[:80]}")
            except Exception as e:
                lines.append(f"  [FAIL] {model}\n         {str(e)[:160]}")
        lines.append("")
    return "\n".join(lines)


@app.get("/metrics")
async def metrics(hours: int = 24):
    return JSONResponse({
        "server": "cloud",
        "version": "2.2.3",
        "models": MODELS,
        "cloud_preset": CLOUD_PRESET,
        "cloud_preset_description": CLOUD_PRESETS[CLOUD_PRESET]["DESCRIPTION"],
        "providers": [
            {
                "name": p.name,
                "base_url": p.base_url,
                "models": p.models,
                "key_present": p.key_present,
            }
            for p in PROVIDERS
        ],
        "providers_configured": [p.name for p in PROVIDERS],
        "providers_skipped": [
            p.name for p in _RAW_PROVIDERS if not (p.key_present and p.models)
        ],
        "rag_enabled": RAG_ENABLED,
        "rag_documents": len(_rag_store.docs) if _rag_store else 0,
        "rag_chunks": len(_rag_store.entries) if _rag_store else 0,
        "rag_doc_ids": sorted(_rag_store.docs.keys()) if _rag_store else [],
        "rag_top_k": RAG_TOP_K if (RAG_ENABLED and _rag_store) else 0,
        "rag_embedder": _rag_embedder.name if _rag_embedder is not None else None,
        "user_dict_enabled": _user_dict is not None,
        "user_dict_size": len(_user_dict.list_words()) if _user_dict is not None else 0,
        "audit": audit.stats(hours=hours),
    })


# ─── /dict/* — user dictionary REST API ──────────────────────────────


@app.get("/dict/list")
async def dict_list():
    if _user_dict is None:
        return JSONResponse(
            {"error": "пользовательский словарь отключён"}, status_code=503,
        )
    return JSONResponse({"words": _user_dict.list_words()})


@app.post("/dict/add")
async def dict_add(request: Request):
    if _user_dict is None:
        return JSONResponse(
            {"error": "пользовательский словарь отключён"}, status_code=503,
        )
    try:
        from shared.user_dict import UserDictError  # noqa: E402
        body = await request.json()
        word = body.get("word") if isinstance(body, dict) else None
        if not isinstance(word, str):
            return JSONResponse({"error": "ожидается JSON-поле 'word'"}, status_code=400)
        added = _user_dict.add(word)
        return JSONResponse({"added": added, "total": len(_user_dict.list_words())})
    except UserDictError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        logger.exception("dict/add failed")
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


@app.post("/dict/remove")
async def dict_remove(request: Request):
    if _user_dict is None:
        return JSONResponse(
            {"error": "пользовательский словарь отключён"}, status_code=503,
        )
    try:
        from shared.user_dict import UserDictError  # noqa: E402
        body = await request.json()
        word = body.get("word") if isinstance(body, dict) else None
        if not isinstance(word, str):
            return JSONResponse({"error": "ожидается JSON-поле 'word'"}, status_code=400)
        removed = _user_dict.remove(word)
        return JSONResponse({"removed": removed, "total": len(_user_dict.list_words())})
    except UserDictError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        logger.exception("dict/remove failed")
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


# ─── /suggest — главный endpoint ─────────────────────────────────────


@app.post("/suggest", response_class=PlainTextResponse)
async def suggest(
    request: Request,
    text: UploadFile = File(...),
    context: UploadFile = File(...),
):
    if _key_missing():
        return "ОШИБКА: OPENROUTER_API_KEY не задан в .env"

    raw_text = (await text.read()).decode("utf-8", errors="replace").strip()
    raw_ctx = (await context.read()).decode("utf-8", errors="replace").strip()
    if not raw_text:
        return "ОШИБКА: Пустой текст"

    # Нормализация переносов строк: исправляет баг «Shift+Enter
    # разрывает текст на разные абзацы». См. _normalize_line_breaks.
    raw_text = _normalize_line_breaks(raw_text)
    raw_ctx = _normalize_line_breaks(raw_ctx)

    extra = _rag_context(raw_text)
    user_msg = f"Контекст (предшествующий текст, только для понимания стиля):\n{raw_ctx}\n"
    if extra:
        user_msg += f"\n{extra}\n"
    user_msg += f"\n---\nТЕКСТ ДЛЯ ПРОВЕРКИ:\n{raw_text}"

    effective_system_prompt = SYSTEM_PROMPT
    if _user_dict is not None:
        dict_suffix = _user_dict.render_for_prompt()
        if dict_suffix:
            effective_system_prompt = SYSTEM_PROMPT + "\n\n" + dict_suffix

    messages = [
        {"role": "system", "content": effective_system_prompt},
        {"role": "user", "content": user_msg},
    ]

    client_ip = request.client.host if request.client else ""
    user_agent = request.headers.get("user-agent", "")

    ok, error, result, used_model = False, "", "", ""
    timer = Timer()
    with timer:
        try:
            result, used_model = await _chat_with_fallback(messages)
            # Cleanup — выравниваем формат до защитных пост-фильтров.
            result = _strip_thinking(result)
            if "===CORRECTED===" not in result:
                result = (
                    "===CORRECTED===\n"
                    f"{result}\n"
                    "===CHANGES===\n"
                    "1. Формат ответа не распознан — проверьте вручную.\n"
                    "===END==="
                )
            if "===END===" not in result:
                result = result.rstrip() + "\n===END==="

            # Минимальный безопасный пост-процессинг. Сетевые модели в целом
            # выдают качественный ответ, поэтому морф-фильтр / sage / LT
            # здесь не используются (см. server/local/main.py для local).
            result = _drop_idempotent_changes(result)        # X → X дроп
            # Сброс ё↔е подстановок в CHANGES (модель иногда «расставляет ё»,
            # хотя это стилистика, не ошибка). Промпт уже это запрещает,
            # но safe-filter закрывает lost-in-the-middle bypass.
            result = _drop_eyo_substitutions(result, raw_text)
            # Посимвольный откат ё→е в CORRECTED закрывает bypass
            # «модель поменяла ё в исправленном тексте, но не вынесла в CHANGES».
            result = _undo_eyo_in_corrected_block(result, raw_text)
            result = _drop_changes_not_in_text(result, raw_text)  # галлюцинации
            result = _renumber_changes(result)               # чистая нумерация
            ok = True
        except OpenRouterError as e:
            error = str(e)
            result = f"ОШИБКА_СЕРВЕРА: {error}"
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"
            logger.exception("Ошибка обработки /suggest")
            result = f"ОШИБКА_СЕРВЕРА: {error}"

    audit.record(
        client_ip=client_ip, user_agent=user_agent,
        server="cloud", model=used_model or "(none)",
        text=raw_text, context=raw_ctx,
        changes_count=count_changes(result),
        duration_ms=timer.ms, ok=ok, error=error,
    )
    logger.info(
        "suggest ip=%s model=%s len=%d ctx=%d changes=%d ok=%s dur=%dms",
        client_ip, used_model, len(raw_text), len(raw_ctx),
        count_changes(result), ok, timer.ms,
    )
    return result
