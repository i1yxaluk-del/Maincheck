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
  russian_quality_models.py \
  syntax_candidates.py \
  safe_diff.py \
  local_rules.py \
  test_hybrid_editor.py \
  test_v31.py \
  test_v4_cascade.py

"$PYTHON_BIN" - <<'PY'
from hybrid_editor import STACKS
print("v4 stacks:", ", ".join(sorted(STACKS)))
PY

SERVICE_USER=${SERVICE_USER:-service}
SERVICE_HOME=$(getent passwd "$SERVICE_USER" 2>/dev/null | cut -d: -f6 || true)
if [ -z "$SERVICE_HOME" ]; then
  SERVICE_USER=$(id -un)
  SERVICE_HOME=${HOME:-$(pwd)}
fi

HF_HOME="$SERVICE_HOME/.cache/huggingface"
mkdir -p "$HF_HOME"
chown -R "$SERVICE_USER":"$SERVICE_USER" "$HF_HOME" 2>/dev/null || true

# SAGE is the fast dedicated Russian spelling/punctuation candidate generator used
# by every preset. Cache it once instead of downloading on the first request.
SAGE_MODEL=${SAGE_CORRECTOR_MODEL:-ai-forever/sage-fredt5-distilled-95m}
if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if id "$SERVICE_USER" >/dev/null 2>&1; then
    su -s /bin/sh "$SERVICE_USER" -c "HF_HOME='$HF_HOME' HUGGINGFACE_HUB_CACHE='$HF_HOME/hub' '$PYTHON_BIN' - <<'PY'
import os
from huggingface_hub import snapshot_download
model = os.environ['SAGE_MODEL']
snapshot_download(model)
print(f'SAGE cached: {model}')
PY" SAGE_MODEL="$SAGE_MODEL"
  else
    HF_HOME="$HF_HOME" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" SAGE_MODEL="$SAGE_MODEL" "$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import snapshot_download
model = os.environ['SAGE_MODEL']
snapshot_download(model)
print(f'SAGE cached: {model}')
PY
  fi
fi

echo "Setup complete. v4 is a multi-candidate proofreading cascade: SAGE + Ollama draft/diagnostic + bounded diff + judge."
