#!/bin/sh
# Install experimental preset Z on Astra Linux 1.8.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"
PYTHON_BIN="$ROOT/venv/bin/python3"
[ -x "$PYTHON_BIN" ] || PYTHON_BIN=python3

export PYTHONPATH="$ROOT:$ROOT/..${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -m py_compile reasoning_cascade.py decision_app_v11.py v10_rules.py
PYTHONPATH="$ROOT:$ROOT/.." "$PYTHON_BIN" -m eval.run_eval --max-fp 0 --min-exact 16

if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: preset Z requires Ollama" >&2
  exit 1
fi
ollama pull deepseek-r1:7b-qwen-distill-q4_K_M
ollama pull hf.co/loqira/Qwen3.5-0.8B-GEC-KAZ-RUS-ENG:Q4_0
cp -n .env.reasoning.example .env || true

echo "Preset Z installed. Review .env, then install ai-suggester.service and restart."
