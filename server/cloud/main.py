"""
Облачный сервер AI LibreOffice Suggester (OpenRouter free tier) — v2.1.

Feature-parity с локальным сервером (`server/local/main.py`):
    • CLOUD_PRESET A/B/C/D — выбор основной модели + fallback-цепочка
      (см. shared/openrouter_client.py).
    • RAG по ведомственным документам (RAG_ENABLED=true) — те же
      data/rag_store/ + Ollama embedder, что и local.
    • Retrieval-augmented few-shot из GEC-банка (USE_FEW_SHOT=true) —
      hybrid BM25 + dense, как в local.
    • Пользовательский словарь + REST API /dict/list, /dict/add, /dict/remove.
    • Морф-фильтр (pymorphy3) — дроп галлюцинированных падежных «улучшений».
    • Морф-детектор — обогащение CHANGES правилами numeral-noun / adj-noun.
    • LanguageTool-RU — стилистика/типографика (опционально).
    • Все пост-фильтры: ё-замена, drop-not-in-text, rebuild-from-diff,
      sync CHANGES↔CORRECTED, renumber.
    • Логи с ротацией + retention, SQLite-аудит, /metrics.

Архитектура: запрос приходит → RAG context + few-shot examples →
OpenRouter chat (fallback по моделям) → весь пост-процессинг shared/postprocess
→ ответ клиенту в формате ===CORRECTED===…===CHANGES===…===END===.

Backward compatibility:
    • Без новых env-флагов поведение совпадает с v1.6.0 (модели обновлены,
      RAG/few-shot/morph/LT — выключены по умолчанию).
    • Старые тесты (`test_cloud_suggest_with_mocked_openrouter`,
      `test_cloud_metrics`, `test_cloud_missing_key`) продолжают работать.
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
    OpenRouterClient,
    OpenRouterError,
    resolve_models,
)
from shared.postprocess import (  # noqa: E402
    _complete_changes_from_corrected,
    _drop_changes_not_in_text,
    _drop_eyo_substitutions,
    _drop_idempotent_changes,
    _drop_morph_case_substitutions,
    _drop_user_dict_changes,
    _enrich_changes_with_detector,
    _enrich_changes_with_languagetool,
    _extract_corrected_body,
    _had_any_change_pairs,
    _has_real_change_items,
    _rebuild_changes_from_diff,
    _renumber_changes,
    _replace_changes_block,
    _strip_thinking,
    _undo_eyo_in_corrected_block,
)


load_dotenv()


# ─── OpenRouter ───────────────────────────────────────────────────────

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT", "90"))
OPENROUTER_TEMPERATURE = float(os.getenv("OPENROUTER_TEMPERATURE", "0.1"))
OPENROUTER_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "3000"))
OPENROUTER_REFERER = os.getenv("OPENROUTER_REFERER", "http://localhost")
OPENROUTER_TITLE = os.getenv("OPENROUTER_TITLE", "AI LibreOffice Suggester")

# CLOUD_PRESET выбирает primary-модель + дефолтный fallback-список.
# OPENROUTER_MODELS (CSV) перебивает preset, если задан.
CLOUD_PRESET = os.getenv("CLOUD_PRESET", "A").strip().upper()
if CLOUD_PRESET not in CLOUD_PRESETS:
    CLOUD_PRESET = "A"
_override_models = [
    s.strip() for s in os.getenv("OPENROUTER_MODELS", "").split(",") if s.strip()
]
MODELS = resolve_models(CLOUD_PRESET, _override_models or None)


# ─── RAG (опционально, RAG_ENABLED=true) ──────────────────────────────

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
RAG_ENABLED = os.getenv("RAG_ENABLED", "false").lower() in ("1", "true", "yes", "on")
RAG_STORE_DIR = os.getenv("RAG_STORE_DIR", "data/rag_store")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
RAG_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")


# ─── Few-shot retrieval (опционально, USE_FEW_SHOT=true) ─────────────

USE_FEW_SHOT = os.getenv("USE_FEW_SHOT", "false").lower() in ("1", "true", "yes", "on")
GEC_BANK_FILES = [
    s.strip() for s in os.getenv(
        "GEC_BANK_FILES",
        "../shared/gec_seed/gec_bank.jsonl"
        ",../shared/gec_seed/gec_bank_extended.jsonl"
        ",../shared/gec_seed/lexify_admin.jsonl",
    ).split(",")
    if s.strip()
]
GEC_TOP_K = int(os.getenv("GEC_TOP_K", "3"))
GEC_EMBED_MODEL = os.getenv("GEC_EMBED_MODEL", RAG_EMBED_MODEL)
GEC_RETRIEVAL_MODE = os.getenv("GEC_RETRIEVAL_MODE", "hybrid").strip().lower()
if GEC_RETRIEVAL_MODE not in ("hybrid", "dense", "sparse"):
    GEC_RETRIEVAL_MODE = "hybrid"
GEC_BM25_TOKENIZER = os.getenv("GEC_BM25_TOKENIZER", "both").strip().lower()
if GEC_BM25_TOKENIZER not in ("word", "trigram", "both"):
    GEC_BM25_TOKENIZER = "both"


# ─── Морфо-фильтр / детектор / словарь / LanguageTool ────────────────

MORPH_FILTER_ENABLED = os.getenv("MORPH_FILTER_ENABLED", "true").lower() in ("1", "true", "yes", "on")
MORPH_DETECTOR_ENABLED = os.getenv("MORPH_DETECTOR_ENABLED", "true").lower() in ("1", "true", "yes", "on")
USER_DICT_ENABLED = os.getenv("USER_DICT_ENABLED", "true").lower() in ("1", "true", "yes", "on")

LANGUAGETOOL_ENABLED = os.getenv("LANGUAGETOOL_ENABLED", "false").lower() in ("1", "true", "yes", "on")
LANGUAGETOOL_URL = os.getenv("LANGUAGETOOL_URL", "http://localhost:8081")
LANGUAGETOOL_LANGUAGE = os.getenv("LANGUAGETOOL_LANGUAGE", "ru-RU")
LANGUAGETOOL_TIMEOUT = float(os.getenv("LANGUAGETOOL_TIMEOUT", "10"))
LANGUAGETOOL_ENABLED_CATEGORIES = os.getenv(
    "LANGUAGETOOL_ENABLED_CATEGORIES", "STYLE,TYPOGRAPHY",
)
LANGUAGETOOL_DISABLED_CATEGORIES = os.getenv("LANGUAGETOOL_DISABLED_CATEGORIES", "")
LANGUAGETOOL_DISABLED_RULES = os.getenv("LANGUAGETOOL_DISABLED_RULES", "")


logger = setup_logger("ai_suggester.cloud")
audit = AuditStore()
or_client = OpenRouterClient(
    OPENROUTER_API_KEY,
    referer=OPENROUTER_REFERER,
    title=OPENROUTER_TITLE,
    timeout=OPENROUTER_TIMEOUT,
)


# ─── Lazy singletons (mirror local/main.py) ───────────────────────────

_rag_store = None
_rag_embedder = None
_gec_bank = None
_morph_filter = None
_morph_detector = None
_user_dict = None
_lt_client = None

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

if USE_FEW_SHOT:
    try:
        from shared.gec_bank import GecBank  # noqa: E402
        from shared.rag_store import HashingEmbedder, OllamaEmbedder  # noqa: E402

        if _rag_embedder is not None and GEC_EMBED_MODEL == RAG_EMBED_MODEL:
            _gec_embedder = _rag_embedder
        else:
            try:
                _gec_embedder = OllamaEmbedder(model=GEC_EMBED_MODEL, base_url=OLLAMA_URL)
                _ = _gec_embedder.embed(["probe"])
            except Exception as exc:
                logger.warning(
                    "Ollama-эмбеддер (%s) недоступен (%s), HashingEmbedder",
                    GEC_EMBED_MODEL, exc,
                )
                _gec_embedder = HashingEmbedder(dim=1024)
        _gec_bank = GecBank(_gec_embedder, bm25_tokenizer=GEC_BM25_TOKENIZER)
        _resolved_paths = [
            str((_HERE / p) if not Path(p).is_absolute() else Path(p))
            for p in GEC_BANK_FILES
        ]
        n = _gec_bank.load_jsonl(*_resolved_paths)
        if n == 0:
            logger.warning("Few-shot retrieval включён, но банк пуст — 0-shot fallback")
            _gec_bank = None
        else:
            _cache_path = Path(_resolved_paths[0]).with_suffix(".index.pkl")
            _gec_bank.build_index(cache_path=_cache_path)
            logger.info(
                "Few-shot retrieval включён: %d пар, top_k=%d, embedder=%s, mode=%s",
                n, GEC_TOP_K, _gec_embedder.name, GEC_RETRIEVAL_MODE,
            )
    except Exception as e:
        logger.warning("Few-shot retrieval не удалось инициализировать: %s", e)
        _gec_bank = None

if MORPH_FILTER_ENABLED:
    try:
        from shared.morph_filter import get_morph_filter  # noqa: E402

        _morph_filter = get_morph_filter()
        if _morph_filter.available:
            logger.info("MorphFilter включён (pymorphy3)")
        else:
            logger.info("MorphFilter: pymorphy3 не доступен — no-op")
    except Exception as e:
        logger.warning("MorphFilter не удалось инициализировать: %s", e)
        _morph_filter = None

if MORPH_DETECTOR_ENABLED:
    try:
        from shared.morph_detector import get_morph_detector  # noqa: E402

        _morph_detector = get_morph_detector()
        if _morph_detector.available:
            logger.info("MorphDetector включён (pymorphy3)")
        else:
            logger.info("MorphDetector: pymorphy3 не доступен — no-op")
    except Exception as e:
        logger.warning("MorphDetector не удалось инициализировать: %s", e)
        _morph_detector = None

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

if LANGUAGETOOL_ENABLED:
    try:
        from shared.languagetool_client import (  # noqa: E402
            _parse_csv_env,
            get_languagetool_client,
        )

        _lt_client = get_languagetool_client(
            url=LANGUAGETOOL_URL,
            language=LANGUAGETOOL_LANGUAGE,
            enabled_categories=_parse_csv_env(LANGUAGETOOL_ENABLED_CATEGORIES),
            disabled_categories=_parse_csv_env(LANGUAGETOOL_DISABLED_CATEGORIES),
            disabled_rules=_parse_csv_env(LANGUAGETOOL_DISABLED_RULES),
            timeout=LANGUAGETOOL_TIMEOUT,
        )
        if _lt_client.available:
            logger.info(
                "LanguageTool включён: url=%s, language=%s",
                LANGUAGETOOL_URL, LANGUAGETOOL_LANGUAGE,
            )
        else:
            logger.warning(
                "LanguageTool: ENABLED=true, но сервер %s недоступен — skip LT-этап",
                LANGUAGETOOL_URL,
            )
    except Exception as e:
        logger.warning("LanguageTool не удалось инициализировать: %s", e)
        _lt_client = None
else:
    logger.info("LanguageTool: отключён (LANGUAGETOOL_ENABLED=false)")


# ─── Промпт идентичен local/main.py v2.0-b ────────────────────────────

SYSTEM_PROMPT = """Ты — корректор русского языка для официальных документов. Не рассуждай, сразу выдавай ответ в нужном формате.

ИСПРАВЛЯЙ ТОЛЬКО ЯВНЫЕ ОШИБКИ:
• орфография — опечатки, удвоение/пропуск букв, слитное/раздельное написание;
• управление — «согласно приказу» (не «согласно приказа»), «благодаря решению»;
• согласование — однородные члены и причастные обороты в одном роде, числе и падеже с главным словом, даже если оно стоит за несколько слов («актов, …, не предусмотренных», не «предусмотренной»);
• пунктуация — запятые при однородных членах, обособленных оборотах, придаточных.

НЕ ТРОГАЙ:
• аббревиатуры и сокращения (п/п, вх.№, исх.№, ФСБ, МВД);
• ведомственные термины и профессиональные обороты;
• правильно написанный текст («улучшать стиль» нельзя);
• е/ё взаимозаменяемы — это стилистика, не ошибка;
• структуру и смысл предложений.

ФОРМАТ ОТВЕТА (строго, без какого-либо текста до или после):
===CORRECTED===
<исправленный текст>
===CHANGES===
1. «было» → «стало» | краткая причина (5–10 слов)
===END===

ПРАВИЛА ЦИТИРОВАНИЯ В CHANGES (важно — клиент применяет правки поиском по тексту):
• в кавычках «было» цитируй ТОЧНО как в исходном тексте, побуквенно;
• БЕЗ многоточия (… или ...), БЕЗ сокращений, БЕЗ перефразирования;
• «стало» — фрагмент той же длины с применённой правкой;
• если правка касается запятой — включи в цитату слова слева и справа от запятой;
• это правило относится к ФОРМАТУ цитаты в CHANGES, а не к тому, какие ошибки искать. Ищи все ошибки правописания, управления, согласования, пунктуации одинаково внимательно.

Если ошибок нет:
===CORRECTED===
<исходный текст без изменений>
===CHANGES===
1. Ошибок не найдено. Текст соответствует нормам.
===END==="""


app = FastAPI(title="AI LibreOffice Suggester — Cloud", version="2.1.0")


# ─── Backward-compatible helpers (тест test_cloud_suggest_with_mocked_openrouter
# подменяет cloud_module.call_model) ──────────────────────────────────


def _key_missing() -> bool:
    return not or_client.key_present


async def call_model(messages: list, model: str) -> str:
    """Один HTTP-запрос на одну модель (backward-compat для тестов).

    Бросает `httpx.HTTPStatusError` на 4xx/5xx (как и в v1.6.0), чтобы
    верхний уровень мог решить, переходить ли к следующей модели по
    fallback-цепочке. Тесты подменяют эту функцию для мокирования.
    """
    return await or_client._post_chat(
        messages, model,
        temperature=OPENROUTER_TEMPERATURE,
        max_tokens=OPENROUTER_MAX_TOKENS,
    )


async def _chat_with_fallback(messages: list) -> tuple[str, str]:
    """Перебирает MODELS до первого успешного ответа. Возвращает (content, used_model).
    На полном фейле — поднимает OpenRouterError.

    Использует `call_model` для каждой попытки — это позволяет тестам
    мокировать одну точку входа.
    """
    last_err = "нет ответа"
    for model in MODELS:
        try:
            content = await call_model(messages, model)
            return content, model
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            last_err = f"[{model}] HTTP {status}: {e.response.text[:160]}"
            if status in (429, 502, 503, 504):
                logger.info("OpenRouter: %s -> HTTP %d, пробую следующую", model, status)
                continue
            logger.warning("OpenRouter: %s -> HTTP %d (не retry)", model, status)
            raise OpenRouterError(last_err) from e
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            last_err = f"[{model}] {type(e).__name__}: {str(e)[:120]}"
            logger.info("OpenRouter: %s timeout/network, пробую следующую", model)
            continue
        except Exception as e:  # noqa: BLE001
            last_err = f"[{model}] {type(e).__name__}: {str(e)[:160]}"
            logger.exception("OpenRouter: непредвиденная ошибка на %s", model)
            continue
    raise OpenRouterError(f"Все модели недоступны. Последняя ошибка: {last_err}")


# ─── /rag_context (shared logic; cloud-local) ─────────────────────────


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
    parts = ["ПРИМЕНИМЫЕ НОРМАТИВНЫЕ ФРАГМЕНТЫ (используйте как справку, не цитируйте в CHANGES):"]
    for h in hits:
        parts.append(f"— [{h['doc_id']}] {h['text'][:600]}")
    return "\n".join(parts)


# ─── Endpoints ────────────────────────────────────────────────────────


@app.get("/health", response_class=PlainTextResponse)
async def health():
    if _key_missing():
        return "ОШИБКА: OPENROUTER_API_KEY не задан в .env"
    probe_msgs = [{"role": "user", "content": "Ответь одним словом: OK"}]
    last_err = "нет ответа"
    for model in MODELS:
        try:
            ans = await call_model(probe_msgs, model)
            extras = []
            if RAG_ENABLED and _rag_store:
                extras.append(f"RAG: {len(_rag_store.docs)} документов")
            if _gec_bank is not None:
                extras.append(f"Few-shot: {len(_gec_bank)} пар, top_k={GEC_TOP_K}")
            suffix = (" | " + " | ".join(extras)) if extras else ""
            return f"OK | Работает: {model} | Ответ: {ans[:40]}{suffix}"
        except httpx.HTTPStatusError as e:
            last_err = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            if e.response.status_code in (429, 502, 503, 504):
                continue
            return f"ОШИБКА HTTP на {model}: {last_err}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
            continue
    return f"ОШИБКА: Все модели недоступны. Последняя ошибка: {last_err}"


@app.get("/test_api", response_class=PlainTextResponse)
async def test_api():
    if _key_missing():
        return "ОШИБКА: OPENROUTER_API_KEY не задан в .env"
    lines = [
        f"Ключ: {or_client.key_redacted}",
        f"Preset: {CLOUD_PRESET} ({CLOUD_PRESETS[CLOUD_PRESET]['DESCRIPTION']})",
        "",
    ]
    for model in MODELS:
        try:
            ans = await call_model(
                [{"role": "user", "content": "Ответь одним словом: OK"}], model,
            )
            lines.append(f"[OK]   {model}\n       → {ans[:80]}")
        except Exception as e:
            lines.append(f"[FAIL] {model}\n       {str(e)[:160]}")
        lines.append("")
    return "\n".join(lines)


@app.get("/metrics")
async def metrics(hours: int = 24):
    return JSONResponse({
        "server": "cloud",
        "models": MODELS,
        "cloud_preset": CLOUD_PRESET,
        "cloud_preset_description": CLOUD_PRESETS[CLOUD_PRESET]["DESCRIPTION"],
        "rag_enabled": RAG_ENABLED,
        "rag_documents": len(_rag_store.docs) if _rag_store else 0,
        "rag_chunks": len(_rag_store.entries) if _rag_store else 0,
        "rag_doc_ids": sorted(_rag_store.docs.keys()) if _rag_store else [],
        "rag_top_k": RAG_TOP_K if (RAG_ENABLED and _rag_store) else 0,
        "rag_embedder": _rag_embedder.name if _rag_embedder is not None else None,
        "few_shot_enabled": _gec_bank is not None,
        "few_shot_pairs": len(_gec_bank) if _gec_bank else 0,
        "few_shot_top_k": GEC_TOP_K if _gec_bank else 0,
        "few_shot_embedder": _gec_bank.embedder.name if _gec_bank is not None else None,
        "few_shot_retrieval_mode": GEC_RETRIEVAL_MODE if _gec_bank is not None else None,
        "few_shot_bm25_terms": (
            _gec_bank.stats().get("bm25_terms", 0) if _gec_bank is not None else 0
        ),
        "morph_filter_enabled": _morph_filter is not None and _morph_filter.available,
        "morph_detector_enabled": _morph_detector is not None and _morph_detector.available,
        "user_dict_enabled": _user_dict is not None,
        "user_dict_size": len(_user_dict.list_words()) if _user_dict is not None else 0,
        "languagetool_enabled": LANGUAGETOOL_ENABLED,
        "languagetool_available": _lt_client is not None and _lt_client.available,
        "languagetool_url": LANGUAGETOOL_URL if LANGUAGETOOL_ENABLED else None,
        "languagetool_language": LANGUAGETOOL_LANGUAGE if LANGUAGETOOL_ENABLED else None,
        "languagetool_enabled_categories": (
            LANGUAGETOOL_ENABLED_CATEGORIES if LANGUAGETOOL_ENABLED else None
        ),
        "audit": audit.stats(hours=hours),
    })


# ─── /dict/* — user dictionary REST API (mirror local/main.py) ───────


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

    # RAG-контекст и few-shot — те же механизмы, что в local-сервере
    extra = _rag_context(raw_text)
    user_msg = f"Контекст (предшествующий текст, только для понимания стиля):\n{raw_ctx}\n"
    if extra:
        user_msg += f"\n{extra}\n"
    user_msg += f"\n---\nТЕКСТ ДЛЯ ПРОВЕРКИ:\n{raw_text}"

    few_shot_examples: list = []
    if _gec_bank is not None:
        try:
            if GEC_RETRIEVAL_MODE == "sparse":
                hits = _gec_bank.search_sparse(raw_text, top_k=GEC_TOP_K)
            elif GEC_RETRIEVAL_MODE == "dense":
                hits = _gec_bank.search(raw_text, top_k=GEC_TOP_K)
            else:
                hits = _gec_bank.search_hybrid(raw_text, top_k=GEC_TOP_K)
            few_shot_examples = [pair for score, pair in hits]
            if few_shot_examples:
                logger.info(
                    "Few-shot (%s): подмешиваю %d пар", GEC_RETRIEVAL_MODE, len(few_shot_examples),
                )
        except Exception as e:
            logger.warning("Few-shot retrieval провалился (0-shot fallback): %s", e)
            few_shot_examples = []

    effective_system_prompt = SYSTEM_PROMPT
    if _user_dict is not None:
        dict_suffix = _user_dict.render_for_prompt()
        if dict_suffix:
            effective_system_prompt = SYSTEM_PROMPT + "\n\n" + dict_suffix

    if few_shot_examples:
        from shared.gec_bank import build_few_shot_messages  # noqa: E402

        messages = build_few_shot_messages(
            system_prompt=effective_system_prompt,
            user_text=user_msg,
            examples=few_shot_examples,
        )
    else:
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
            # Cleanup — выравниваем формат до того, как пускать через пост-фильтры
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

            # Pipeline идентичен local/main.py (см. ЖУРНАЛ_v1.6.md):
            had_pairs_pre_filter = _had_any_change_pairs(result)
            result = _drop_idempotent_changes(result)
            result = _drop_eyo_substitutions(result, raw_text)
            result = _undo_eyo_in_corrected_block(result, raw_text)
            result = _drop_morph_case_substitutions(result, raw_text, _morph_filter)
            result = _drop_changes_not_in_text(result, raw_text)
            if had_pairs_pre_filter and not _has_real_change_items(result):
                corrected_body = _extract_corrected_body(result)
                if corrected_body and corrected_body != raw_text:
                    rebuilt = _rebuild_changes_from_diff(raw_text, corrected_body)
                    if rebuilt:
                        logger.info(
                            "Реконструировано %d пункт(ов) CHANGES из diff "
                            "(модель забыла рапорт)", len(rebuilt),
                        )
                        result = _replace_changes_block(result, rebuilt)
            result = _drop_user_dict_changes(result, _user_dict)
            result = _enrich_changes_with_detector(
                result, raw_text, _morph_detector, user_dict=_user_dict,
            )
            result = _enrich_changes_with_languagetool(
                result, raw_text, _lt_client, user_dict=_user_dict,
            )
            result = _complete_changes_from_corrected(result, raw_text)
            result = _renumber_changes(result)
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
