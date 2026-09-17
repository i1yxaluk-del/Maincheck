#!/usr/bin/env bash
set -uo pipefail
ROOT=${MAINCHK_ROOT:-/home/service/llama}; PROFILES=${*:-main openvino llama-json}; STAMP=$(date +%Y%m%d-%H%M%S); OUT="$ROOT/results/v20-$STAMP"; READY=/run/maincheck-v20-ready; STATUS=/run/maincheck-v20-install.status; CORPUS="$ROOT/server/v20/cases.jsonl"; mkdir -p "$OUT"
wait_ready(){ local p=$1; for _ in $(seq 1 240); do [[ -f "$READY" && "$(cat "$READY")" == "$p" ]] && curl --max-time 5 -fsS http://127.0.0.1:8000/health >/dev/null && return 0; [[ -f "$STATUS" ]] && grep -q '^failed ' "$STATUS" && { cat "$STATUS" >&2; return 1; }; sleep 5; done; echo "profile $p readiness timeout" >&2; return 1; }
unavailable(){ printf '%s\tunavailable\t-\t-\t-\t-\t-\t-\t0\n' "$1" | tee -a "$OUT/summary.tsv"; }
printf 'profile\tpass_rate\tpositive_recall\tclean_preservation\tfalse_positive_rate\tlayout_preservation\tmedian_ms\tp95_ms\tcases\n' | tee "$OUT/summary.tsv"
for p in $PROFILES; do
  if ! sudo -E "$ROOT/scripts/v20/install.sh" "$p" || ! wait_ready "$p"; then unavailable "$p"; continue; fi
  if ! PYTHONPATH="$ROOT/server" "$ROOT/server/local/venv/bin/python" -m v20.benchmark --url http://127.0.0.1:8000 --corpus "$CORPUS" --out "$OUT/$p.json"; then unavailable "$p"; continue; fi
  "$ROOT/server/local/venv/bin/python" - "$p" "$OUT/$p.json" <<'PY' | tee -a "$OUT/summary.tsv"
import json,sys
p=sys.argv[1];d=json.load(open(sys.argv[2],encoding='utf-8'));print('\t'.join(map(str,[p,d['pass_rate'],d['positive_recall'],d['clean_preservation'],d['false_positive_rate'],d['layout_preservation'],d['median_ms'],d['p95_ms'],d['cases']])))
PY
done
echo "V20 BENCHMARK COMPLETE results=$OUT"; cat "$OUT/summary.tsv"
