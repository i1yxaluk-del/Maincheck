#!/usr/bin/env bash
set -euo pipefail
ROOT=${MAINCHK_ROOT:-/home/service/llama};URL=${V20_URL:-http://127.0.0.1:8000};PY=${V20_PYTHON:-$ROOT/server/local/venv/bin/python};CORPUS=${V20_CORPUS:-$ROOT/server/v20/cases-current-30.jsonl};OUT_DIR=${V20_RESULTS_DIR:-$ROOT/results};STAMP=$(date +%Y%m%d-%H%M%S);OUT="$OUT_DIR/current-30-$STAMP.json"
mkdir -p "$OUT_DIR"
HEALTH=$(curl --max-time 10 -fsS "$URL/health")
if [[ ${BENCHMARK_REQUIRE_STACK_Z:-true} == true && "$HEALTH" != *'stack=Z'* ]]; then echo "ERROR: expected stack=Z; health=$HEALTH" >&2;exit 2;fi
PYTHONPATH="$ROOT/server" "$PY" -m v20.benchmark --url "$URL" --corpus "$CORPUS" --out "$OUT" >/dev/null
"$PY" - "$OUT" <<'PY'
import json,sys
p=sys.argv[1];d=json.load(open(p,encoding='utf-8'))
for k in ('cases','passed','pass_rate','positive_recall','clean_preservation','false_positive_rate','layout_preservation','reason_quality','internal_reason_leaks','errors','median_ms','p95_ms'):print(f'{k}={d[k]}')
print('by_category:')
for k,v in d['by_category'].items():print(f"  {k}: {v}")
failed=[r['id'] for r in d['rows'] if not r['pass']];print('failed='+(','.join(failed) if failed else 'none'))
print(f'V20 CURRENT BENCHMARK COMPLETE results={p}')
PY
