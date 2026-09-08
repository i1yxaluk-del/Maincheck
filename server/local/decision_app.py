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
from pipelines import STACKS, StackRouter
from shared.audit import AuditStore, Timer, count_changes
from shared.logging_setup import setup_logger

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
LLM_PRESET = os.getenv("LLM_PRESET", "A").strip().upper()
if LLM_PRESET not in STACKS:
    raise RuntimeError(f"Unsupported LLM_PRESET={LLM_PRESET!r}; expected A, B, C, F or G")

MIN_CONFIDENCE = float(os.getenv("DECISION_MIN_CONFIDENCE", "0.60"))
MAX_CHANGES = int(os.getenv("DECISION_MAX_CHANGES", "12"))
MAX_BEFORE_CHARS = int(os.getenv("DECISION_MAX_BEFORE_CHARS", "120"))
USER_DICT_ENABLED = os.getenv("USER_DICT_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
WARMUP = os.getenv("OLLAMA_WARMUP", "true").lower() in {"1", "true", "yes", "on"}
AUDIT_ENABLED = os.getenv("AUDIT_ENABLED", "true").lower() in {"1", "true", "yes", "on"}

logger = setup_logger("ai_suggester.local")
audit = AuditStore() if AUDIT_ENABLED else None

try:
    from shared.morph_detector import get_morph_detector

    morph_detector = get_morph_detector() if os.getenv("MORPH_DETECTOR_ENABLED", "true").lower() in {"1", "true", "yes", "on"} else None
except Exception as exc:
    logger.warning("MorphDetector unavailable: %s", exc)
    morph_detector = None

try:
    from shared.user_dict import get_user_dict

    user_dict = get_user_dict() if USER_DICT_ENABLED else None
except Exception as exc:
    logger.warning("UserDict unavailable: %s", exc)
    user_dict = None

router = StackRouter(LLM_PRESET, morph_detector)
app = FastAPI(title="AI LibreOffice Suggester", version="2.4")


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
            f"{i}. «{c.before}» → «{c.after}» | {c.reason or 'явная ошибка'}"
            for i, c in enumerate(accepted, 1)
        )
    else:
        changes = "1. Ошибок не найдено."
    return f"===CORRECTED===\n{corrected}\n===CHANGES===\n{changes}\n===END==="


def dict_words() -> set[str]:
    if user_dict is None:
        return set()
    try:
        return set(user_dict.list_words())
    except Exception:
        return set()


@app.on_event("startup")
async def startup() -> None:
    logger.info(
        "Stack=%s (%s), model=%s, experimental=%s, retrieval=%s",
        router.info.name,
        router.info.description,
        router.info.model,
        router.info.experimental,
        router.retriever.available,
    )
    if WARMUP:
        started = time.perf_counter()
        try:
            await router.warmup()
            logger.info("Warmup OK in %d ms", int((time.perf_counter() - started) * 1000))
        except Exception as exc:
            logger.warning("Warmup failed: %s", exc)


@app.get("/health", response_class=PlainTextResponse)
async def health() -> str:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()
        return f"OK | stack={router.info.name} | model={router.info.model}"
    except Exception as exc:
        return f"DEGRADED | stack={router.info.name} | Ollama error: {exc}"


@app.get("/metrics")
async def metrics(hours: int = 24):
    data = router.metrics()
    return JSONResponse({
        "server": "local",
        "version": "2.4",
        "stack": router.info.name,
        "description": router.info.description,
        "model": router.info.model,
        "experimental": router.info.experimental,
        "morph_detector_available": bool(morph_detector and getattr(morph_detector, "available", False)),
        "user_dict_enabled": user_dict is not None,
        "user_dict_size": len(dict_words()),
        "audit": audit.stats(hours=hours) if audit is not None else {"enabled": False},
        **data,
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
    candidates, accepted = [], []
    ok, error = True, ""
    try:
        with timer:
            candidates = await router.candidates(raw_text, raw_ctx, dict_words())
            engine = DecisionEngine(
                min_confidence=MIN_CONFIDENCE,
                max_changes=MAX_CHANGES,
                max_before_chars=MAX_BEFORE_CHARS,
                protected_words=dict_words(),
            )
            corrected, accepted = engine.apply(raw_text, candidates)
            result = render_result(corrected, accepted)
    except Exception as exc:
        ok = False
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("Suggestion failed")
        result = f"ОШИБКА_СЕРВЕРА: {error}"

    logger.info(
        "suggest stack=%s len=%d ctx=%d candidates=%d accepted=%d dur=%dms",
        router.info.name, len(raw_text), len(raw_ctx), len(candidates), len(accepted), timer.ms,
    )

    if audit is not None:
        audit.record(
            client_ip=request.client.host if request.client else "",
            user_agent=request.headers.get("user-agent", ""),
            server="local", model=router.info.model, text=raw_text, context=raw_ctx,
            changes_count=count_changes(result), duration_ms=timer.ms, ok=ok, error=error,
        )
    return result
