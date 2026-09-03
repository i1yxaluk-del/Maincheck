#!/bin/bash
set -euo pipefail

# Production is managed by systemd. This helper never starts a second uvicorn
# process; it only ensures the configured Ollama model exists and restarts the
# existing ai-suggester.service.
cd "$(dirname "$0")"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is required" >&2
  exit 1
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

MODEL="${MODEL_NAME:-qwen3.5:4b}"
if ! ollama list | awk '{print $1}' | grep -Fxq "$MODEL"; then
  echo "Pulling $MODEL..."
  ollama pull "$MODEL"
fi

echo "Restarting systemd service ai-suggester.service..."
sudo systemctl restart ai-suggester.service
sudo systemctl --no-pager --full status ai-suggester.service
