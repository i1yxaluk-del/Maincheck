#!/usr/bin/env bash
set -euo pipefail
ROOT=${MAINCHK_ROOT:-/home/service/llama};URL=${V20_URL:-http://127.0.0.1:8000};PY=${V20_PYTHON:-$ROOT/server/local/venv/bin/python};CORPUS=${V20_CORPUS:-$ROOT/server/v20/cases-deep-40.jsonl};OUT_DIR=${V20_RESULTS_DIR:-$ROOT/results};STAMP=$(date +%Y%m%d-%H%M%S);OUT="$OUT_DIR/deep-40-$STAMP.json"
mkdir -p "$OUT_DIR"
if [[ ! -f "$CORPUS" || ${V20_REBUILD_CORPUS:-true} == true ]];then "$PY" "$ROOT/scripts/v20/build_deep_corpus_v2.py" --out "$CORPUS";fi
HEALTH=$(curl --max-time 10 -fsS "$URL/health")
if [[ ${BENCHMARK_REQUIRE_STACK_Z:-true} == true && "$HEALTH" != *'stack=Z'* ]]; then echo "ERROR: expected stack=Z; health=$HEALTH" >&2;exit 2;fi
PYTHONPATH="$ROOT/server" "$PY" -m v20.benchmark --url "$URL" --corpus "$CORPUS" --out "$OUT" >/dev/null
"$PY" - "$OUT" <<'PY'
import json,sys
p=sys.argv[1];d=json.load(open(p,encoding='utf-8'))
for k in ('cases','passed','pass_rate','positive_recall','clean_preservation','false_positive_rate','exact_match_rate','delimiter_preservation','layout_preservation','reason_quality','internal_reason_leaks','errors','median_ms','p95_ms'):print(f'{k}={d[k]}')
print('by_category:')
for k,v in d['by_category'].items():print(f"  {k}: {v}")
failed=[r['id'] for r in d['rows'] if not r['pass']];print('failed='+(','.join(failed) if failed else 'none'))
print(f'V20 DEEP BENCHMARK COMPLETE results={p}')
PY
