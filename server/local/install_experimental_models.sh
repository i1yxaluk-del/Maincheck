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
  decision_app.py decision_engine.py hybrid_editor.py qwen35_backend.py \
  russian_gec_backend.py russian_quality_models.py safe_diff.py syntax_candidates.py \
  local_rules.py test_hybrid_editor.py test_v31.py test_v5_cascade.py

"$PYTHON_BIN" - <<'PY'
from hybrid_editor import STACKS
print("v5 stacks:", ", ".join(sorted(STACKS)))
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

SAGE_MODEL=${SAGE_CORRECTOR_MODEL:-ai-forever/sage-fredt5-distilled-95m}
GEC_BASE=${RUSSIAN_GEC_BASE_MODEL:-Qwen/Qwen3.5-0.8B}
GEC_ADAPTER=${RUSSIAN_GEC_ADAPTER_REPO:-synterr-nlp/bea2026-gec-adapters}
GEC_SUBFOLDER=${RUSSIAN_GEC_ADAPTER_SUBFOLDER:-v4_qwen35_08b_lorugec}

export SAGE_MODEL GEC_BASE GEC_ADAPTER GEC_SUBFOLDER

run_hf_cache() {
  if id "$SERVICE_USER" >/dev/null 2>&1; then
    su -s /bin/sh "$SERVICE_USER" -c "HF_HOME='$HF_HOME' HUGGINGFACE_HUB_CACHE='$HF_HOME/hub' SAGE_MODEL='$SAGE_MODEL' GEC_BASE='$GEC_BASE' GEC_ADAPTER='$GEC_ADAPTER' GEC_SUBFOLDER='$GEC_SUBFOLDER' '$PYTHON_BIN' - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(os.environ['SAGE_MODEL'])
snapshot_download(os.environ['GEC_BASE'])
snapshot_download(os.environ['GEC_ADAPTER'], allow_patterns=[os.environ['GEC_SUBFOLDER'] + '/*'])
print('Cached:', os.environ['SAGE_MODEL'])
print('Cached:', os.environ['GEC_BASE'])
print('Cached:', os.environ['GEC_ADAPTER'] + ':' + os.environ['GEC_SUBFOLDER'])
PY"
  else
    HF_HOME="$HF_HOME" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" \
      "$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(os.environ['SAGE_MODEL'])
snapshot_download(os.environ['GEC_BASE'])
snapshot_download(os.environ['GEC_ADAPTER'], allow_patterns=[os.environ['GEC_SUBFOLDER'] + '/*'])
print('Cached:', os.environ['SAGE_MODEL'])
print('Cached:', os.environ['GEC_BASE'])
print('Cached:', os.environ['GEC_ADAPTER'] + ':' + os.environ['GEC_SUBFOLDER'])
PY
  fi
}

run_hf_cache

echo "Setup complete: v5 fast SAGE + measured Russian GEC + Ollama full-draft cascade."
