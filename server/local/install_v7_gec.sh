#!/bin/sh
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
GEC_MODEL=${OLLAMA_GEC_MODEL:-hf.co/loqira/Qwen3.5-0.8B-GEC-KAZ-RUS-ENG:Q4_0}

if [ -x "$ROOT/venv/bin/pip" ]; then
  "$ROOT/venv/bin/pip" install -r requirements.txt
fi

export PYTHONPATH="$ROOT/..${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -m py_compile decision_app.py hybrid_editor.py ollama_gec.py russian_quality_models.py safe_diff.py

if id "$SERVICE_USER" >/dev/null 2>&1; then
  su -s /bin/sh "$SERVICE_USER" -c "HF_HOME='$HF_HOME' '$PYTHON_BIN' -c 'from huggingface_hub import snapshot_download; snapshot_download(\"$SAGE_MODEL\")'"
else
  HF_HOME="$HF_HOME" "$PYTHON_BIN" -c "from huggingface_hub import snapshot_download; snapshot_download('$SAGE_MODEL')"
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: ollama executable not found" >&2
  exit 1
fi

echo "Pulling $GEC_MODEL"
ollama pull "$GEC_MODEL"

echo "v7 install complete: SAGE=$SAGE_MODEL GEC=$GEC_MODEL"
