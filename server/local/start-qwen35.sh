#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is required" >&2
  exit 1
fi
MODEL="${MODEL_NAME:-qwen3.5:4b}"
if ! ollama list | awk '{print $1}' | grep -Fxq "$MODEL"; then
  echo "Pulling $MODEL..."
  ollama pull "$MODEL"
fi
export OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
export MODEL_NAME="$MODEL"
export NUM_THREADS="${NUM_THREADS:-28}"
export OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX:-4096}"
export OLLAMA_NUM_PREDICT="${OLLAMA_NUM_PREDICT:-512}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-24h}"
export OLLAMA_THINK="${OLLAMA_THINK:-false}"
export OLLAMA_TEMPERATURE="${OLLAMA_TEMPERATURE:-0}"
export USE_FEW_SHOT="${USE_FEW_SHOT:-true}"
export GEC_TOP_K="${GEC_TOP_K:-1}"
export GEC_RETRIEVAL_MODE="${GEC_RETRIEVAL_MODE:-hybrid}"
export GEC_EMBED_MODEL="${GEC_EMBED_MODEL:-bge-m3}"
export DECISION_ENGINE_ENABLED="${DECISION_ENGINE_ENABLED:-true}"
export DECISION_MIN_CONFIDENCE="${DECISION_MIN_CONFIDENCE:-0.55}"
export DECISION_MAX_CHANGES="${DECISION_MAX_CHANGES:-40}"
export DECISION_MAX_BEFORE_CHARS="${DECISION_MAX_BEFORE_CHARS:-180}"
export MORPH_FILTER_ENABLED="${MORPH_FILTER_ENABLED:-true}"
export MORPH_DETECTOR_ENABLED="${MORPH_DETECTOR_ENABLED:-false}"
export LANGUAGETOOL_ENABLED="${LANGUAGETOOL_ENABLED:-false}"
export USER_DICT_ENABLED="${USER_DICT_ENABLED:-true}"
export RAG_ENABLED="${RAG_ENABLED:-false}"
exec uvicorn decision_app:app --host 0.0.0.0 --port 8000
