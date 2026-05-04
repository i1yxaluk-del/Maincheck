"""
Облачный сервер AI LibreOffice Suggester (OpenRouter free tier).

Новое в v1.3:
    • Логи с ротацией и retention
    • SQLite-аудит запросов (/metrics)
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

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

MODELS = [
    "openrouter/free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "nvidia/llama-3.1-nemotron-nano-8b-v1:free",
]

logger = setup_logger("ai_suggester.cloud")
audit = AuditStore()

SYSTEM_PROMPT = """Ты — корректор русского языка для официальных документов. Не рассуждай, сразу выдавай ответ в нужном формате.

ИСПРАВЛЯЙ ТОЛЬКО ЯВНЫЕ ОШИБКИ:
• орфография — опечатки, удвоение/пропуск букв, слитное/раздельное написание;
• управление — «согласно приказу» (не «согласно приказа»), «благодаря решению»;
• согласование — однородные члены в одном падеже и числе с главным словом;
• пунктуация — запятые при однородных членах, обособленных оборотах, придаточных.

НЕ ТРОГАЙ:
• аббревиатуры и сокращения (п/п, вх.№, исх.№, ФСБ, МВД);
• ведомственные термины и профессиональные обороты;
• правильно написанный текст («улучшать стиль» нельзя);
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

app = FastAPI(title="AI LibreOffice Suggester — Cloud", version="1.6.0")


def _key_missing() -> bool:
    return not OPENROUTER_API_KEY or "ваш_ключ" in OPENROUTER_API_KEY.lower()


async def call_model(messages: list, model: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "LibreOffice AI Suggester",
    }
    async with httpx.AsyncClient(timeout=90.0) as client:
        r = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={"model": model, "messages": messages, "temperature": 0.1, "max_tokens": 3000},
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


@app.get("/health", response_class=PlainTextResponse)
async def health():
    if _key_missing():
        return "ОШИБКА: OPENROUTER_API_KEY не задан в .env"
    probe = [{"role": "user", "content": "Ответь одним словом: OK"}]
    for model in MODELS:
        try:
            ans = await call_model(probe, model)
            return f"OK | Работает: {model} | Ответ: {ans[:40]}"
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                continue
            return f"ОШИБКА HTTP {e.response.status_code} на {model}: {e.response.text[:200]}"
        except Exception as e:
            return f"ОШИБКА на {model}: {str(e)[:200]}"
    return "ОШИБКА: Все модели вернули 429 (rate limit). Попробуйте позже."


@app.get("/test_api", response_class=PlainTextResponse)
async def test_api():
    if _key_missing():
        return "ОШИБКА: OPENROUTER_API_KEY не задан в .env"
    lines = [f"Ключ: {OPENROUTER_API_KEY[:12]}...{OPENROUTER_API_KEY[-4:]}\n"]
    for model in MODELS:
        try:
            ans = await call_model([{"role": "user", "content": "Ответь одним словом: OK"}], model)
            lines.append(f"[OK]   {model}\n       → {ans[:80]}")
        except Exception as e:
            lines.append(f"[FAIL] {model}\n       {str(e)[:120]}")
        lines.append("")
    return "\n".join(lines)


@app.get("/metrics")
async def metrics(hours: int = 24):
    return JSONResponse({
        "server": "cloud",
        "models": MODELS,
        "audit": audit.stats(hours=hours),
    })


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

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Контекст (предшествующий текст, только для понимания стиля):\n{raw_ctx}\n\n"
            f"---\nТЕКСТ ДЛЯ ПРОВЕРКИ:\n{raw_text}"
        )},
    ]

    client_ip = request.client.host if request.client else ""
    user_agent = request.headers.get("user-agent", "")

    last_err = "нет ответа"
    used_model = ""
    ok, error, result = False, "", ""
    timer = Timer()
    with timer:
        for model in MODELS:
            try:
                result = await call_model(messages, model)
                used_model = model
                if "===CORRECTED===" not in result:
                    result = (
                        "===CORRECTED===\n"
                        f"{result}\n"
                        "===CHANGES===\n"
                        "1. Формат ответа не распознан — проверьте вручную.\n"
                        "===END==="
                    )
                # Гарантия терминатора (см. server/local/main.py): дописываем
                # ===END=== если модель его не отдала. Без этого клиент
                # на не-2xx HTTP считает ответ обрезанным.
                if "===END===" not in result:
                    result = result.rstrip() + "\n===END==="
                ok = True
                break
            except httpx.HTTPStatusError as e:
                last_err = f"[{model}] HTTP {e.response.status_code}"
                if e.response.status_code in (429, 502, 503):
                    continue
                error = last_err
                result = f"ОШИБКА_СЕРВЕРА: {last_err}"
                break
            except Exception as e:
                last_err = f"[{model}] {str(e)[:200]}"
                continue
        if not ok and not result:
            error = last_err
            result = f"ОШИБКА_СЕРВЕРА: Все модели недоступны. {last_err}"

    audit.record(
        client_ip=client_ip, user_agent=user_agent,
        server="cloud", model=used_model or "(none)",
        text=raw_text, context=raw_ctx,
        changes_count=count_changes(result),
        duration_ms=timer.ms, ok=ok, error=error,
    )
    logger.info(
        "suggest ip=%s model=%s len=%d changes=%d ok=%s dur=%dms",
        client_ip, used_model, len(raw_text),
        count_changes(result), ok, timer.ms,
    )
    return result
