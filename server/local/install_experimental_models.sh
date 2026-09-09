#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"

if [ -x "$ROOT/venv/bin/pip" ]; then
  "$ROOT/venv/bin/pip" install -r requirements.txt
else
  python3 -m pip install -r requirements.txt
fi

PYTHON_BIN="$ROOT/venv/bin/python3"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN=python3
fi

export PYTHONPATH="$ROOT/..${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m py_compile \
  decision_app.py \
  decision_engine.py \
  hybrid_editor.py \
  qwen35_backend.py \
  syntax_candidates.py \
  safe_diff.py \
  local_rules.py \
  test_hybrid_editor.py \
  test_v31.py

"$PYTHON_BIN" - <<'PY'
from hybrid_editor import STACKS
print("v3.1 stacks:", ", ".join(sorted(STACKS)))
PY

# X/Y now use the compact GGUF specialist through the same Ollama daemon as A/B.
# Do not download the old 4B Transformers checkpoint into the FastAPI host.
PRESET=${LLM_PRESET:-A}
MODEL_ID=${GEC_SPECIALIST_MODEL:-hf.co/loqira/Qwen3.5-0.8B-GEC-KAZ-RUS-ENG:Q4_0}
if [ "$PRESET" = "X" ] || [ "$PRESET" = "Y" ]; then
  if command -v ollama >/dev/null 2>&1; then
    echo "Pulling compact X/Y specialist into Ollama: $MODEL_ID"
    ollama pull "$MODEL_ID"
  else
    echo "WARNING: ollama CLI not found; X/Y will pull the specialist lazily through OLLAMA_URL." >&2
  fi
fi

echo "Setup complete. v3.1 presets: A (production), B (production-candidate), X/Y (experimental)."
