#!/usr/bin/env bash
set -euo pipefail
ROOT=${MAINCHK_ROOT:-/home/service/llama}; PROFILES=${*:-main openvino llama-json}; STAMP=$(date +%Y%m%d-%H%M%S); OUT="$ROOT/results/v20-$STAMP"; mkdir -p "$OUT"
wait_ready(){ local p=$1; for _ in $(seq 1 240); do curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1 && return 0; sleep 5; done; echo "profile $p health timeout" >&2; return 1; }
printf 'profile\tpass_rate\tmedian_ms\tp95_ms\tcases\n' | tee "$OUT/summary.tsv"
for p in $PROFILES; do
  sudo "$ROOT/scripts/v20/install.sh" "$p"
  wait_ready "$p"
  PYTHONPATH="$ROOT/server" "$ROOT/server/local/venv/bin/python" -m v20.benchmark --url http://127.0.0.1:8000 --corpus "$ROOT/tests/v20/cases.jsonl" --out "$OUT/$p.json"
  "$ROOT/server/local/venv/bin/python" - "$p" "$OUT/$p.json" <<'PY' | tee -a "$OUT/summary.tsv"
import json,sys
p=sys.argv[1]; d=json.load(open(sys.argv[2],encoding='utf-8')); print(f"{p}\t{d['pass_rate']}\t{d['median_ms']}\t{d['p95_ms']}\t{d['cases']}")
PY
done
echo "V20 BENCHMARK COMPLETE results=$OUT"
cat "$OUT/summary.tsv"
