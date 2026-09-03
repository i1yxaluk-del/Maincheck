from __future__ import annotations

import json
import os

import httpx

import main as legacy
from decision_engine import DecisionEngine

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3.5:4b")
NUM_THREADS = int(os.getenv("NUM_THREADS", "28"))
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))
TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))
TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "24h")
THINK = os.getenv("OLLAMA_THINK", "false").lower() in ("1", "true", "yes", "on")

SCHEMA = {
    "type": "object",
    "properties": {
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "before": {"type": "string"},
                    "after": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "category": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["before", "after", "confidence", "category", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["edits"],
    "additionalProperties": False,
}

SYSTEM = """Ты — строгий корректор русского официально-делового текста.
Исправляй только реальные языковые ошибки: орфографию, явные опечатки,
пунктуацию, согласование и управление. Не переписывай стиль, не улучшай
формулировки, не меняй допустимые падежи, термины, аббревиатуры, имена,
названия организаций или юридические обозначения. Не нормализуй ё/е.

Верни ТОЛЬКО JSON по заданной схеме. Каждая правка должна содержать точный
фрагмент BEFORE из исходного текста и AFTER. Если сомневаешься — не предлагай
правку. confidence отражает уверенность именно в необходимости изменения.
"""


def _extract_text(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = str(msg.get("content", ""))
        marker = "ТЕКСТ ДЛЯ ПРОВЕРКИ:\n"
        if marker in content:
            return content.split(marker, 1)[1].strip().removesuffix("\n\n/no_think")
    return ""


def _render(corrected: str, accepted) -> str:
    lines = [f"{i}. {c.before} → {c.after}" for i, c in enumerate(accepted, 1)]
    changes = "\n".join(lines) if lines else "Ошибок не найдено."
    return f"===CORRECTED===\n{corrected}\n===CHANGES===\n{changes}\n===END==="


async def decision_call_ollama(messages: list) -> str:
    raw_text = _extract_text(messages)
    protected: set[str] = set()
    if getattr(legacy, "_user_dict", None) is not None:
        try:
            protected = set(legacy._user_dict.list_words())
        except Exception:
            pass

    # Qwen3.5 is sensitive to duplicated system prompts. Keep one system
    # message and preserve retrieved user/assistant few-shot turns.
    user_history = [m for m in messages if m.get("role") != "system"]
    prompt_messages = [{"role": "system", "content": SYSTEM}, *user_history]
    if not prompt_messages or prompt_messages[-1].get("role") != "user":
        prompt_messages.append({"role": "user", "content": raw_text})
    if prompt_messages and prompt_messages[-1].get("role") == "user" and not prompt_messages[-1].get("content", "").rstrip().endswith("/no_think"):
        prompt_messages[-1]["content"] = prompt_messages[-1]["content"].rstrip() + "\n\n/no_think"

    payload = {
        "model": MODEL_NAME,
        "messages": prompt_messages,
        "stream": False,
        "format": SCHEMA,
        "think": THINK,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": TEMPERATURE,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "num_thread": NUM_THREADS,
            "repeat_penalty": 1.05,
        },
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "{}").strip()

    try:
        candidates = DecisionEngine.parse(content)
    except (ValueError, TypeError, json.JSONDecodeError):
        candidates = []

    engine = DecisionEngine(
        min_confidence=float(os.getenv("DECISION_MIN_CONFIDENCE", "0.55")),
        max_changes=int(os.getenv("DECISION_MAX_CHANGES", "40")),
        max_before_chars=int(os.getenv("DECISION_MAX_BEFORE_CHARS", "180")),
        protected_words=protected,
    )
    corrected, accepted = engine.apply(raw_text, candidates)
    logger = getattr(legacy, "logger", None)
    if logger is not None:
        logger.info("DecisionEngine: candidates=%d accepted=%d", len(candidates), len(accepted))
    return _render(corrected, accepted)


# Keep the existing FastAPI routes, audit, post-filters and LibreOffice protocol.
# The systemd entrypoint points to this module, so there is still exactly one
# uvicorn process on :8000; main.py remains the legacy application core.
legacy.call_ollama = decision_call_ollama
app = legacy.app
