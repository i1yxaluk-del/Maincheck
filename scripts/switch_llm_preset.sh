#!/usr/bin/env bash
# Switch one of the three supported local stacks.
#   A — production T-lite structured edits
#   F — experimental Spell-Corrector-RU-4B surface gate
#   G — experimental MorphDetector + T-lite verifier
set -euo pipefail

PRESET="${1:-}"
ENV_FILE="${2:-server/local/.env}"

case "$PRESET" in
  A|a)
    PRESET=A
    MODEL_TAG="t-tech/T-lite-it-2.1:q4_K_M"
    DESC="production: T-lite structured edits"
    ;;
  F|f)
    PRESET=F
    MODEL_TAG=""
    DESC="experimental: Spell-Corrector-RU-4B surface gate"
    ;;
  G|g)
    PRESET=G
    MODEL_TAG="t-tech/T-lite-it-2.1:q4_K_M"
    DESC="experimental: MorphDetector + T-lite verifier"
    ;;
  *)
    echo "Usage: $0 <A|F|G> [env-file]" >&2
    exit 2
    ;;
esac

echo "→ Preset $PRESET: $DESC"

if [ -n "$MODEL_TAG" ]; then
  if ! command -v ollama >/dev/null 2>&1; then
    echo "ОШИБКА: ollama CLI не найден" >&2
    exit 1
  fi
  if ! ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -qFx "$MODEL_TAG"; then
    echo "→ T-lite не найден локально, загружаю..."
    ollama pull "$MODEL_TAG"
  fi
fi

mkdir -p "$(dirname "$ENV_FILE")"
touch "$ENV_FILE"
TMP="$(mktemp)"
grep -vE '^[[:space:]]*(#[[:space:]]*)?LLM_PRESET=' "$ENV_FILE" | grep -vE '^[[:space:]]*MODEL_NAME=' > "$TMP" || true
mv "$TMP" "$ENV_FILE"
printf 'LLM_PRESET=%s\n' "$PRESET" >> "$ENV_FILE"

echo "ГОТОВО. Применить: sudo systemctl restart ai-suggester.service"
echo "Проверить: curl -s http://localhost:8000/metrics"
