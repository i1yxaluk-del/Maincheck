#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is required" >&2
  exit 1
fi
MODEL="qwen3.5:9b"
if ! ollama list | awk '{print $1}' | grep -Fxq "$MODEL"; then
  echo "Pulling $MODEL..."
  ollama pull "$MODEL"
fi
export OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
export MODEL_NAME="${MODEL_NAME:-$MODEL}"
export NUM_THREADS="${NUM_THREADS:-28}"
export OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX:-8192}"
export OLLAMA_NUM_PREDICT="${OLLAMA_NUM_PREDICT:-768}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-24h}"
export OLLAMA_THINK="${OLLAMA_THINK:-false}"
export OLLAMA_TEMPERATURE="${OLLAMA_TEMPERATURE:-0.05}"
export USE_FEW_SHOT="${USE_FEW_SHOT:-true}"
export GEC_TOP_K="${GEC_TOP_K:-1}"
export GEC_RETRIEVAL_MODE="${GEC_RETRIEVAL_MODE:-hybrid}"
export GEC_EMBED_MODEL="${GEC_EMBED_MODEL:-bge-m3}"
export MORPH_FILTER_ENABLED="${MORPH_FILTER_ENABLED:-true}"
export MORPH_DETECTOR_ENABLED="${MORPH_DETECTOR_ENABLED:-false}"
export LANGUAGETOOL_ENABLED="${LANGUAGETOOL_ENABLED:-false}"
exec uvicorn main:app --host 0.0.0.0 --port 8000
