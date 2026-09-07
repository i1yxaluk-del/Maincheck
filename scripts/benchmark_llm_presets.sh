#!/usr/bin/env bash
# Benchmark A/F/G on the same text and context.
set -euo pipefail

TEXT_FILE="${1:-}"
CTX_FILE="${2:-/dev/null}"
if [[ -z "$TEXT_FILE" || ! -f "$TEXT_FILE" ]]; then
  echo "Usage: $0 <text.txt> [ctx.txt]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUGGEST_URL="${SUGGEST_URL:-http://localhost:8000/suggest}"
HEALTH_URL="${HEALTH_URL:-http://localhost:8000/health}"
METRICS_URL="${METRICS_URL:-http://localhost:8000/metrics}"

declare -A LATENCIES=()
declare -A MODELS=()

for PRESET in A F G; do
  echo "=== Preset $PRESET ==="
  "$SCRIPT_DIR/switch_llm_preset.sh" "$PRESET"
  sudo systemctl restart ai-suggester.service

  for i in $(seq 1 180); do
    sleep 1
    if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then break; fi
    [[ "$i" -eq 180 ]] && { echo "warmup timeout for $PRESET" >&2; continue 2; }
  done

  MODELS[$PRESET]="$(curl -fsS "$METRICS_URL" | python3 -c "import json,sys; print(json.load(sys.stdin).get('model','?'))")"
  OUT="/tmp/bench_preset_${PRESET}.txt"
  START=$(date +%s.%N)
  if [[ "$CTX_FILE" != "/dev/null" ]]; then
    curl -fsS --max-time 300 -X POST "$SUGGEST_URL" -F "text=@${TEXT_FILE}" -F "context=@${CTX_FILE}" -o "$OUT"
  else
    curl -fsS --max-time 300 -X POST "$SUGGEST_URL" -F "text=@${TEXT_FILE}" -F "context=@/dev/null" -o "$OUT"
  fi
  END=$(date +%s.%N)
  LATENCIES[$PRESET]="$(python3 - <<PY
start=float("$START"); end=float("$END"); print(f"{end-start:.2f}s")
PY
)"
  echo "latency=${LATENCIES[$PRESET]} output=$OUT"
done

echo
echo "Preset  Latency   Model"
for PRESET in A F G; do
  printf "%-7s %-9s %s\n" "$PRESET" "${LATENCIES[$PRESET]:-FAIL}" "${MODELS[$PRESET]:-?}"
done
