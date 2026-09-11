from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from decision_engine import DecisionEngine
from hybrid_editor import STACKS, HybridRouter
from shared.audit import AuditStore, Timer, count_changes
from shared.logging_setup import setup_logger
from verification import GenerativeGuard

SERVER_VERSION = "9.0"

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
LLM_PRESET = os.getenv("LLM_PRESET", "A").strip().upper()
if LLM_PRESET not in STACKS:
    raise RuntimeError(f"Unsupported LLM_PRESET={LLM_PRESET!r}; expected A, B, X or Y")

MIN_CONFIDENCE = float(os.getenv("DECISION_MIN_CONFIDENCE", "0.55"))
MAX_CHANGES = int(os.getenv("DECISION_MAX_CHANGES", "12"))
MAX_BEFORE_CHARS = int(os.getenv("DECISION_MAX_BEFORE_CHARS", "120"))
USER_DICT_ENABLED = os.getenv("USER_DICT_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
WARMUP = os.getenv("OLLAMA_WARMUP", "true").lower() in {"1", "true", "yes", "on"}
AUDIT_ENABLED = os.getenv("AUDIT_ENABLED", "true").lower() in {"1", "true", "yes", "on"}

logger = setup_logger("ai_suggester.local")
audit = AuditStore() if AUDIT_ENABLED else None

try:
    from shared.user_dict import get_user_dict
    user_dict = get_user_dict() if USER_DICT_ENABLED else None
except Exception as exc:
    logger.warning("UserDict unavailable: %s", exc)
    user_dict = None


def dict_words() -> set[str]:
    if user_dict is None:
        return set()
    try:
        return set(user_dict.list_words())
    except Exception:
        return set()


router = HybridRouter(LLM_PRESET, dict_words())
guard = GenerativeGuard(protected_words=dict_words())
app = FastAPI(title="AI LibreOffice Suggester", version=SERVER_VERSION)


def build_engine() -> DecisionEngine:
    """Собирает решающий движок с фильтром галлюцинаций.

    До v9 `GenerativeGuard`-эквивалент (`_is_unverified_llm_inflection`)
    проверял только категории `model*`/`unknown*`, а реальные категории
    стека называются `sage-spell-punc`, `russian-gec`, `draft_*` — то
    есть антигаллюцинационная защита в проде не работала вовсе.
    """
    return DecisionEngine(
        min_confidence=MIN_CONFIDENCE,
        max_changes=MAX_CHANGES,
        max_before_chars=MAX_BEFORE_CHARS,
        protected_words=dict_words(),
        guard=guard,
    )


def normalize_line_breaks(text: str) -> str:
    if not text:
        return text
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u2028", "\n")
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text


def render_result(corrected: str, accepted) -> str:
    if accepted:
        changes = "\n".join(
            f"{i}. «{c.before}» → «{c.after}» | {c.reason or 'объективная ошибка'}"
            for i, c in enumerate(accepted, 1)
        )
    else:
        changes = "1. Ошибок не найдено."
    return f"===CORRECTED===\n{corrected}\n===CHANGES===\n{changes}\n===END==="


@app.on_event("startup")
async def startup() -> None:
    logger.info(
        "Stack=%s (%s), generator=%s, experimental=%s, SAGE=%s, GEC=%s, "
        "retrieval=%s, rules=%s, spellcheck=%s, languagetool=%s, rescue=%s",
        router.info.name,
        router.info.description,
        router.info.model,
        router.info.experimental,
        router.sage.model_id if router.sage.available else "disabled",
        router.gec.model if router.gec.available else "disabled",
        router.retriever.count if router.retriever.available else 0,
        router.rules.available,
        router.speller.available,
        router.languagetool.url if router.languagetool.available else "disabled",
        router.rescue_mode,
    )
    if WARMUP:
        started = time.perf_counter()
        await router.warmup()
        elapsed = int((time.perf_counter() - started) * 1000)
        if router.degraded:
            logger.warning("Warmup completed degraded in %d ms: %s", elapsed, router.degraded)
        else:
            logger.info("Warmup OK in %d ms", elapsed)


@app.on_event("shutdown")
async def shutdown() -> None:
    await router.aclose()


@app.get("/health", response_class=PlainTextResponse)
async def health() -> str:
    detail = (
        f"stack={router.info.name} | model={router.gec.model} | "
        f"rules={router.rules.available} | spellcheck={router.speller.available}"
    )
    if not router.ollama_required():
        state = "DEGRADED" if router.degraded else "OK"
        return f"{state} | {detail} | ollama=not required"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()
    except Exception as exc:
        # Детерминированные стадии работают и без Ollama, поэтому это
        # деградация, а не отказ: клиент по-прежнему получает правки.
        return f"DEGRADED | {detail} | Ollama error: {exc}"
    state = "DEGRADED" if router.degraded else "OK"
    return f"{state} | {detail} | degraded={len(router.degraded)}"


@app.get("/metrics")
async def metrics(hours: int = 24):
    return JSONResponse({
        "server": "local",
        "version": SERVER_VERSION,
        "stack": router.info.name,
        "description": router.info.description,
        "model": router.info.model,
        "experimental": router.info.experimental,
        "rule_engine_available": router.rules.available,
        "retrieval_count": router.retriever.count,
        "sage": router.sage.metrics().__dict__,
        "russian_gec": router.gec.metrics().__dict__,
        "languagetool": router.languagetool.metrics().__dict__,
        "guard_rejections": guard.rejections,
        "user_dict_enabled": user_dict is not None,
        "user_dict_size": len(dict_words()),
        "audit": audit.stats(hours=hours) if audit is not None else {"enabled": False},
        **router.metrics(),
    })


@app.get("/dict/list")
async def dict_list():
    if user_dict is None:
        return JSONResponse({"error": "пользовательский словарь отключён"}, status_code=503)
    return JSONResponse({"words": sorted(dict_words(), key=str.casefold)})


@app.post("/dict/add")
async def dict_add(request: Request):
    if user_dict is None:
        return JSONResponse({"error": "пользовательский словарь отключён"}, status_code=503)
    body = await request.json()
    word = body.get("word") if isinstance(body, dict) else None
    if not isinstance(word, str) or not word.strip():
        return JSONResponse({"error": "ожидается JSON-поле 'word'"}, status_code=400)
    try:
        added = user_dict.add(word)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({"added": added, "total": len(dict_words())})


@app.post("/dict/remove")
async def dict_remove(request: Request):
    if user_dict is None:
        return JSONResponse({"error": "пользовательский словарь отключён"}, status_code=503)
    body = await request.json()
    word = body.get("word") if isinstance(body, dict) else None
    if not isinstance(word, str) or not word.strip():
        return JSONResponse({"error": "ожидается JSON-поле 'word'"}, status_code=400)
    try:
        removed = user_dict.remove(word)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({"removed": removed, "total": len(dict_words())})


@app.post("/suggest", response_class=PlainTextResponse)
async def suggest(request: Request, text: UploadFile = File(...), context: UploadFile = File(...)):
    raw_text = normalize_line_breaks((await text.read()).decode("utf-8", errors="replace").strip())
    raw_ctx = normalize_line_breaks((await context.read()).decode("utf-8", errors="replace").strip())
    if not raw_text:
        return "ОШИБКА: Пустой текст"

    timer = Timer()
    candidates, accepted, rejections = [], [], []
    ok, error = True, ""
    try:
        with timer:
            candidates = await router.candidates(raw_text, raw_ctx)
            engine = build_engine()
            corrected, accepted = engine.apply(raw_text, candidates)
            rejections = engine.rejections
            result = render_result(corrected, accepted)
    except Exception as exc:
        ok = False
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("Suggestion failed")
        result = f"ОШИБКА_СЕРВЕРА: {error}"

    metrics = router.metrics()
    logger.info(
        "suggest v9 stack=%s len=%d ctx=%d candidates=%d accepted=%d rejected=%d "
        "sources=%s stages=%s stage_ms=%s dur=%dms degraded=%d",
        router.info.name,
        len(raw_text),
        len(raw_ctx),
        len(candidates),
        len(accepted),
        len(rejections),
        sorted({c.category for c in accepted}),
        metrics.get("stage_calls"),
        metrics.get("stage_ms"),
        timer.ms,
        len(router.degraded),
    )
    for candidate, reason in rejections[:8]:
        logger.debug(
            "rejected %r -> %r (%s): %s",
            candidate.before, candidate.after, candidate.category, reason,
        )

    if audit is not None:
        audit.record(
            client_ip=request.client.host if request.client else "",
            user_agent=request.headers.get("user-agent", ""),
            server="local", model=router.info.model, text=raw_text, context=raw_ctx,
            changes_count=count_changes(result), duration_ms=timer.ms, ok=ok, error=error,
        )
    return result
