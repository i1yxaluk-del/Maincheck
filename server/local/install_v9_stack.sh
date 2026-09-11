#!/bin/sh
# Install unified local stack on Astra Linux 1.8.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"
PYTHON_BIN="$ROOT/venv/bin/python3"
[ -x "$PYTHON_BIN" ] || PYTHON_BIN=python3
SERVICE_USER=${SERVICE_USER:-service}
SERVICE_HOME=$(getent passwd "$SERVICE_USER" 2>/dev/null | cut -d: -f6 || true)
[ -n "$SERVICE_HOME" ] || SERVICE_HOME=${HOME:-$(pwd)}
HF_HOME="$SERVICE_HOME/.cache/huggingface"
mkdir -p "$HF_HOME"
SAGE_MODEL=${SAGE_CORRECTOR_MODEL:-ai-forever/sage-fredt5-distilled-95m}
RUPUNCT_MODEL=${RUPUNCT_MODEL:-RUPunct/RUPunct_big}
GEC_MODEL=${OLLAMA_GEC_MODEL:-hf.co/loqira/Qwen3.5-0.8B-GEC-KAZ-RUS-ENG:Q4_0}

[ ! -x "$ROOT/venv/bin/pip" ] || "$ROOT/venv/bin/pip" install -r requirements.txt
export PYTHONPATH="$ROOT/..${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -m py_compile decision_app_v11.py rupunct_stage.py punctuation_pipeline.py reasoning_cascade.py
PYTHONPATH="$ROOT:$ROOT/.." "$PYTHON_BIN" -m eval.run_eval --max-fp 0 --min-exact 16

DOWNLOAD="from huggingface_hub import snapshot_download; snapshot_download('$SAGE_MODEL'); snapshot_download('$RUPUNCT_MODEL')"
if id "$SERVICE_USER" >/dev/null 2>&1; then
  su -s /bin/sh "$SERVICE_USER" -c "HF_HOME='$HF_HOME' '$PYTHON_BIN' -c \"$DOWNLOAD\""
else
  HF_HOME="$HF_HOME" "$PYTHON_BIN" -c "$DOWNLOAD"
fi
if command -v ollama >/dev/null 2>&1; then
  ollama pull "$GEC_MODEL"
else
  echo "WARNING: Ollama not found; deterministic, SAGE and RuPunct remain available" >&2
fi
echo "Install complete: SAGE=$SAGE_MODEL RUPUNCT=$RUPUNCT_MODEL GEC=$GEC_MODEL"
