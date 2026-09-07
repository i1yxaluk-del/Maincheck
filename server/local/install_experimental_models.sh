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

# Fail before touching systemd if the Python tree or HF runtime is broken.
"$PYTHON_BIN" -m py_compile \
  decision_app.py \
  decision_engine.py \
  main.py \
  experimental_backend.py \
  experimental_backend/__init__.py \
  test_experimental_backend.py

"$PYTHON_BIN" - <<'PY'
import peft
import transformers
import experimental_backend
from transformers import Qwen3_5ForCausalLM
from experimental_backend import ExperimentalRouter
print(f"HF stack: transformers={transformers.__version__} peft={peft.__version__}")
print("Qwen3.5 text architecture: OK")
print(f"Experimental backend import: {experimental_backend.__file__}")
print(f"ExperimentalRouter: {ExperimentalRouter.__name__}")
PY

# The systemd service runs as 'service' on the production host. When this
# installer is invoked with sudo, downloading as root would put the HF cache
# in /root and the service would download the models again on first startup.
SERVICE_USER=${SERVICE_USER:-service}
SERVICE_HOME=$(getent passwd "$SERVICE_USER" 2>/dev/null | cut -d: -f6 || true)
if [ -z "$SERVICE_HOME" ]; then
  SERVICE_USER=$(id -un)
  SERVICE_HOME=${HOME:-$(pwd)}
fi

HF_HOME="$SERVICE_HOME/.cache/huggingface"
mkdir -p "$HF_HOME"
chown -R "$SERVICE_USER":"$SERVICE_USER" "$HF_HOME" 2>/dev/null || true

echo "Caching Hugging Face models for service user: $SERVICE_USER"

if id "$SERVICE_USER" >/dev/null 2>&1; then
  su -s /bin/sh "$SERVICE_USER" -c "HF_HOME='$HF_HOME' HUGGINGFACE_HUB_CACHE='$HF_HOME/hub' '$PYTHON_BIN' - <<'PY'
from huggingface_hub import snapshot_download

print('Downloading D base model: Qwen/Qwen3.5-4B')
snapshot_download('Qwen/Qwen3.5-4B')
print('Downloading D adapter: synterr-nlp/bea2026-gec-adapters')
snapshot_download('synterr-nlp/bea2026-gec-adapters')
print('Downloading F model: melsmm/Spell-Corrector-RU-4B')
snapshot_download('melsmm/Spell-Corrector-RU-4B')
print('Experimental HF models cached.')
PY"
else
  HF_HOME="$HF_HOME" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" "$PYTHON_BIN" - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download('Qwen/Qwen3.5-4B')
snapshot_download('synterr-nlp/bea2026-gec-adapters')
snapshot_download('melsmm/Spell-Corrector-RU-4B')
print('Experimental HF models cached.')
PY
fi

# C uses the existing Ollama secondary model. Pull it once when Ollama is
# available; failure is non-fatal because C's startup check reports it.
if command -v ollama >/dev/null 2>&1; then
  ollama pull 'hf.co/loqira/Qwen3.5-0.8B-GEC-KAZ-RUS-ENG:Q4_0' || true
fi

echo 'One-time preset setup complete. Switch only LLM_PRESET=A/C/D/E/F/G and restart ai-suggester.service.'
