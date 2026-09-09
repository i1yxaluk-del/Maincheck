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

SERVICE_USER=${SERVICE_USER:-service}
SERVICE_HOME=$(getent passwd "$SERVICE_USER" 2>/dev/null | cut -d: -f6 || true)
if [ -z "$SERVICE_HOME" ]; then
  SERVICE_USER=$(id -un)
  SERVICE_HOME=${HOME:-$(pwd)}
fi

HF_HOME="$SERVICE_HOME/.cache/huggingface"
mkdir -p "$HF_HOME"
chown -R "$SERVICE_USER":"$SERVICE_USER" "$HF_HOME" 2>/dev/null || true

# X/Y use a text-only Russian GEC specialist. This deliberately avoids the
# previous Qwen3.5 multimodal processor and its Pillow/Torchvision dependency.
PRESET=${LLM_PRESET:-A}
if [ "$PRESET" = "X" ] || [ "$PRESET" = "Y" ]; then
  MODEL_ID=${GEC_SPECIALIST_MODEL:-ReginaNasyrova/checkpoint_150_lora_grpo_upd_reward_GECExplanation-4B-sft-stage1-March2026}
  if id "$SERVICE_USER" >/dev/null 2>&1; then
    su -s /bin/sh "$SERVICE_USER" -c "HF_HOME='$HF_HOME' HUGGINGFACE_HUB_CACHE='$HF_HOME/hub' PYTHONPATH='$ROOT/..${PYTHONPATH:+:$PYTHONPATH}' GEC_SPECIALIST_MODEL='$MODEL_ID' '$PYTHON_BIN' - <<'PY'
import os
from huggingface_hub import snapshot_download
model = os.environ['GEC_SPECIALIST_MODEL']
snapshot_download(model)
print(f'GEC specialist cached: {model}')
PY"
  else
    HF_HOME="$HF_HOME" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" GEC_SPECIALIST_MODEL="$MODEL_ID" "$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import snapshot_download
model = os.environ['GEC_SPECIALIST_MODEL']
snapshot_download(model)
print(f'GEC specialist cached: {model}')
PY
  fi
fi

echo "Setup complete. v3.1 presets: A (production), B (production-candidate), X/Y (experimental)."
