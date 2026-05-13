#!/usr/bin/env bash
# v2.0-a: A/B/C benchmark — гонит один и тот же текст через каждый
# preset и сравнивает latency + CHANGES блоки.
#
# Использование:
#   ./scripts/benchmark_llm_presets.sh /path/to/text.txt [/path/to/ctx.txt]
#
# Что делает:
#   1. Для каждого preset A/B/C:
#      a) запускает switch_llm_preset.sh (пуллит модель если нужно)
#      b) перезапускает сервер
#      c) ждёт warmup (через GET /health)
#      d) делает POST /suggest, замеряет время
#      e) сохраняет ответ в /tmp/bench_<preset>.txt
#   2. Печатает сводную таблицу latency + diff между preset-ами
#
# Требует sudo для systemctl restart. Если нет — запустите вручную
# `sudo systemctl restart ai-suggester` между запусками.

set -euo pipefail

TEXT_FILE="${1:-}"
CTX_FILE="${2:-/dev/null}"

if [[ -z "$TEXT_FILE" || ! -f "$TEXT_FILE" ]]; then
  echo "Usage: $0 <text.txt> [ctx.txt]" >&2
  echo "" >&2
  echo "Пример (нужен файл с тестовым текстом):" >&2
  echo "  echo 'Проверочное мероприятия по контролю.' > /tmp/text.txt" >&2
  echo "  $0 /tmp/text.txt" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUGGEST_URL="${SUGGEST_URL:-http://localhost:8000/suggest}"
HEALTH_URL="${HEALTH_URL:-http://localhost:8000/health}"
METRICS_URL="${METRICS_URL:-http://localhost:8000/metrics}"

declare -A LATENCIES=()
declare -A MODELS=()

for PRESET in A B C; do
  echo ""
  echo "═══════════════════════════════════════════════════════════"
  echo "  Preset $PRESET"
  echo "═══════════════════════════════════════════════════════════"

  "$SCRIPT_DIR/switch_llm_preset.sh" "$PRESET" || {
    echo "ОШИБКА: preset $PRESET — пропускаем." >&2
    LATENCIES[$PRESET]="FAIL (pull)"
    continue
  }

  echo "→ Перезапуск ai-suggester..."
  if ! sudo systemctl restart ai-suggester 2>/dev/null; then
    echo "ВНИМАНИЕ: sudo systemctl restart не сработал."
    echo "Перезапустите сервер вручную и нажмите Enter:"
    read -r _
  fi

  echo "→ Ждём warmup (до 6 минут)..."
  for i in $(seq 1 360); do
    sleep 1
    if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
      echo "  ready (через ${i} с)"
      break
    fi
    if [[ $((i % 30)) -eq 0 ]]; then
      echo "  ...ещё ждём (${i} с)"
    fi
    if [[ $i -eq 360 ]]; then
      echo "ОШИБКА: сервер не поднялся за 6 минут, пропускаем preset $PRESET" >&2
      LATENCIES[$PRESET]="FAIL (warmup)"
      continue 2
    fi
  done

  # Захватим имя модели для отчёта
  MODELS[$PRESET]="$(curl -fsS "$METRICS_URL" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('model','?'))" 2>/dev/null || echo '?')"

  echo "→ Запрос: POST $SUGGEST_URL"
  OUT="/tmp/bench_preset_${PRESET}.txt"
  START=$(date +%s.%N)
  if [[ "$CTX_FILE" != "/dev/null" ]]; then
    curl -fsS --max-time 600 -X POST "$SUGGEST_URL" \
      -F "text=@${TEXT_FILE}" -F "context=@${CTX_FILE}" -o "$OUT"
  else
    curl -fsS --max-time 600 -X POST "$SUGGEST_URL" \
      -F "text=@${TEXT_FILE}" -o "$OUT"
  fi
  END=$(date +%s.%N)
  DUR=$(printf "%.1f" "$(echo "$END - $START" | bc -l)")
  LATENCIES[$PRESET]="${DUR} s"
  echo "→ Готово (${DUR} с), результат: $OUT"
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Сводная таблица"
echo "═══════════════════════════════════════════════════════════"
echo ""
printf "%-10s %-12s %s\n" "Preset" "Latency" "Model"
printf "%-10s %-12s %s\n" "------" "-------" "-----"
for PRESET in A B C; do
  printf "%-10s %-12s %s\n" "$PRESET" "${LATENCIES[$PRESET]:-skip}" "${MODELS[$PRESET]:-?}"
done

echo ""
echo "CHANGES блоки сохранены:"
for PRESET in A B C; do
  if [[ -f "/tmp/bench_preset_${PRESET}.txt" ]]; then
    echo "  /tmp/bench_preset_${PRESET}.txt"
  fi
done

echo ""
echo "Сравнение CHANGES (diff A→B, A→C):"
if [[ -f /tmp/bench_preset_A.txt && -f /tmp/bench_preset_B.txt ]]; then
  echo "--- A vs B ---"
  diff /tmp/bench_preset_A.txt /tmp/bench_preset_B.txt | head -40 || true
fi
if [[ -f /tmp/bench_preset_A.txt && -f /tmp/bench_preset_C.txt ]]; then
  echo "--- A vs C ---"
  diff /tmp/bench_preset_A.txt /tmp/bench_preset_C.txt | head -40 || true
fi
