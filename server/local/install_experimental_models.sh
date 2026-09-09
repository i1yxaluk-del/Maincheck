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
  test_hybrid_editor.py

"$PYTHON_BIN" - <<'PY'
from hybrid_editor import STACKS
print("v3 stacks:", ", ".join(sorted(STACKS)))
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

# X/Y are the only presets requiring the official Qwen3.5-4B weights.
# A/B use the already deployed Ollama models and do not download anything.
PRESET=${LLM_PRESET:-A}
if [ "$PRESET" = "X" ] || [ "$PRESET" = "Y" ]; then
  if id "$SERVICE_USER" >/dev/null 2>&1; then
    su -s /bin/sh "$SERVICE_USER" -c "HF_HOME='$HF_HOME' HUGGINGFACE_HUB_CACHE='$HF_HOME/hub' PYTHONPATH='$ROOT/..${PYTHONPATH:+:$PYTHONPATH}' '$PYTHON_BIN' - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-4B')
print('Qwen3.5-4B cached.')
PY"
  else
    HF_HOME="$HF_HOME" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" "$PYTHON_BIN" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-4B')
print('Qwen3.5-4B cached.')
PY
  fi
fi

echo "Setup complete. v3 presets: A (production), B (production-candidate), X/Y (experimental)."
