#!/bin/sh
# Install preset Z using the unified .env.v9.example configuration.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"
PYTHON_BIN="$ROOT/venv/bin/python3"
[ -x "$PYTHON_BIN" ] || PYTHON_BIN=python3

export PYTHONPATH="$ROOT:$ROOT/..${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -m py_compile reasoning_cascade.py punctuation_pipeline.py decision_app_v11.py v10_rules.py
PYTHONPATH="$ROOT:$ROOT/.." "$PYTHON_BIN" -m eval.run_eval --max-fp 0 --min-exact 16

command -v ollama >/dev/null 2>&1 || { echo "ERROR: preset Z requires Ollama" >&2; exit 1; }
ollama pull deepseek-r1:7b-qwen-distill-q4_K_M
ollama pull hf.co/loqira/Qwen3.5-0.8B-GEC-KAZ-RUS-ENG:Q4_0

if [ ! -f .env ]; then
  cp .env.v9.example .env
  sed -i 's/^LLM_PRESET=.*/LLM_PRESET=Z/' .env
fi

echo "Preset Z installed in the unified .env."
echo "For punctuation install local LanguageTool on 127.0.0.1:8081."
