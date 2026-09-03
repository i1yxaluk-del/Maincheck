from __future__ import annotations

import json
import os

# A/B/C is the single experiment switch. The preset wins over any stale
# MODEL_NAME left in the .env so a restart is enough to change stacks.
LLM_PRESET = os.getenv("LLM_PRESET", "A").strip().upper()
PRESETS = {
    "A": {
        "model": "t-tech/T-lite-it-2.1:q4_K_M",
        "description": "T-lite baseline",
        "secondary": False,
    },
    "B": {
        "model": "qwen3.5:4b",
        "description": "Qwen3.5-4B + DecisionEngine",
        "secondary": False,
    },
    "C": {
        "model": "t-tech/T-lite-it-2.1:q4_K_M",
        "description": "Hybrid T-lite + compact surface GEC",
        "secondary": True,
    },
}
if LLM_PRESET not in PRESETS:
    raise RuntimeError(f"Unsupported LLM_PRESET={LLM_PRESET!r}; expected A, B or C")

STACK = PRESETS[LLM_PRESET]
# main.py reads MODEL_NAME during import and also uses it for warmup/metrics.
os.environ["MODEL_NAME"] = STACK["model"]
os.environ["LLM_PRESET"] = LLM_PRESET
if STACK["secondary"]:
    os.environ["SECONDARY_GEC_ENABLED"] = "true"
    os.environ.setdefault(
        "SECONDARY_GEC_MODEL",
        "hf.co/loqira/Qwen3.5-0.8B-GEC-KAZ-RUS-ENG:Q4_0",
    )
else:
    os.environ["SECONDARY_GEC_ENABLED"] = "false"

import httpx

import main as legacy
from decision_engine import DecisionEngine
from secondary_gec import SecondaryGEC, SecondaryEdit

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = STACK["model"]
NUM_THREADS = int(os.getenv("NUM_THREADS", "28"))
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))
TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))
TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "24h")
THINK = os.getenv("OLLAMA_THINK", "false").lower() in ("1", "true", "yes", "on")
SECONDARY = (
    SecondaryGEC(
        model=os.getenv("SECONDARY_GEC_MODEL", "hf.co/loqira/Qwen3.5-0.8B-GEC-KAZ-RUS-ENG:Q4_0"),
        timeout=float(os.getenv("SECONDARY_GEC_TIMEOUT", "90")),
        keep_alive=os.getenv("SECONDARY_GEC_KEEP_ALIVE", "5m"),
        max_edits=int(os.getenv("SECONDARY_GEC_MAX_EDITS", "4")),
    )
    if STACK["secondary"]
    else None
)

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


def _render(corrected: str, accepted, secondary_edits: list[SecondaryEdit]) -> str:
    lines = [f"{i}. {c.before} → {c.after}" for i, c in enumerate(accepted, 1)]
    start = len(lines) + 1
    lines.extend(
        f"{i}. {e.before} → {e.after}"
        for i, e in enumerate(secondary_edits, start)
    )
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
    secondary_edits: list[SecondaryEdit] = []
    if SECONDARY is not None:
        corrected, secondary_edits = await SECONDARY.enrich(corrected)

    logger = getattr(legacy, "logger", None)
    if logger is not None:
        logger.info(
            "Preset=%s (%s) model=%s DecisionEngine: candidates=%d accepted=%d secondary=%d",
            LLM_PRESET,
            STACK["description"],
            MODEL_NAME,
            len(candidates),
            len(accepted),
            len(secondary_edits),
        )
    return _render(corrected, accepted, secondary_edits)


async def _startup_secondary_check() -> None:
    if SECONDARY is None:
        return
    await SECONDARY.check_available()


# Keep the single existing FastAPI application/process. This adds only a startup
# diagnostic for preset C; it never downloads models automatically.
app = legacy.app
app.add_event_handler("startup", _startup_secondary_check)
legacy.call_ollama = decision_call_ollama
