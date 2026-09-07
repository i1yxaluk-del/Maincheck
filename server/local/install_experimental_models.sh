#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"

# Install the complete runtime once. After this, switching A/C/D/E/F/G in
# .env requires only a service restart; no per-preset pip install is needed.
if [ -x "$ROOT/venv/bin/pip" ]; then
  "$ROOT/venv/bin/pip" install -r requirements.txt
else
  python3 -m pip install -r requirements.txt
fi

PYTHON_BIN="$ROOT/venv/bin/python3"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN=python3
fi

"$PYTHON_BIN" - <<'PY'
from huggingface_hub import snapshot_download

print('Downloading D base model: Qwen/Qwen3.5-4B')
snapshot_download('Qwen/Qwen3.5-4B')
print('Downloading D adapter: synterr-nlp/bea2026-gec-adapters')
snapshot_download('synterr-nlp/bea2026-gec-adapters')
print('Downloading F model: melsmm/Spell-Corrector-RU-4B')
snapshot_download('melsmm/Spell-Corrector-RU-4B')
print('Experimental models cached. Presets A/C/D/E/F/G can now be switched only via LLM_PRESET + service restart.')
PY
