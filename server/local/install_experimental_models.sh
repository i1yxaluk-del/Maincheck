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

"$PYTHON_BIN" -m py_compile \
  decision_app.py \
  decision_engine.py \
  pipelines.py \
  test_experimental_backend.py

"$PYTHON_BIN" - <<'PY'
import transformers
from pipelines import STACKS
print(f"Transformers: {transformers.__version__}")
print("Stacks:", ", ".join(sorted(STACKS)))
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

if id "$SERVICE_USER" >/dev/null 2>&1; then
  su -s /bin/sh "$SERVICE_USER" -c "HF_HOME='$HF_HOME' HUGGINGFACE_HUB_CACHE='$HF_HOME/hub' '$PYTHON_BIN' - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('melsmm/Spell-Corrector-RU-4B')
print('F model cached.')
PY"
else
  HF_HOME="$HF_HOME" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" "$PYTHON_BIN" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('melsmm/Spell-Corrector-RU-4B')
print('F model cached.')
PY
fi

echo 'Setup complete. Supported stacks: A (production), F/G (experimental).'
