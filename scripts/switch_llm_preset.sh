#!/usr/bin/env bash
# v2.0-a: быстрое переключение LLM preset (A/B/C) для A/B-тестирования.
# Использование:
#   ./scripts/switch_llm_preset.sh A   # T-lite-it-2.1 (default)
#   ./scripts/switch_llm_preset.sh B   # YandexGPT-5-Lite-8B
#   ./scripts/switch_llm_preset.sh C   # GigaChat-3.1-Lightning
#
# Что делает:
#   1. Проверяет что preset валиден
#   2. Пуллит модель через ollama (если ещё не загружена)
#   3. Обновляет .env: устанавливает LLM_PRESET=X, убирает MODEL_NAME override
#   4. Подсказывает перезапустить сервер
#
# НЕ перезапускает сервер сам — это решение оставляем оператору.

set -euo pipefail

PRESET="${1:-}"
ENV_FILE="${2:-server/local/.env}"

case "$PRESET" in
  A|a)
    PRESET=A
    MODEL_TAG="t-tech/T-lite-it-2.1:q4_K_M"
    DESC="T-lite-it-2.1 8B (baseline, T-Bank Qwen3-fine-tune)"
    ;;
  B|b)
    PRESET=B
    MODEL_TAG="hf.co/yandex/YandexGPT-5-Lite-8B-instruct-GGUF:Q4_K_M"
    DESC="YandexGPT-5-Lite-8B-instruct (F0.5=83% LORuGEC, ACL BEA 2025)"
    ;;
  C|c)
    PRESET=C
    MODEL_TAG="hf.co/ai-sage/GigaChat-3.1-Lightning-10B-A1.8B-Instruct-GGUF:Q4_K_M"
    DESC="GigaChat-3.1-Lightning-10B-A1.8B (MoE 1.8B active, MIT)"
    ;;
  *)
    echo "Usage: $0 <A|B|C> [env-file]" >&2
    echo "" >&2
    echo "  A — t-tech/T-lite-it-2.1:q4_K_M (baseline, default)" >&2
    echo "  B — hf.co/yandex/YandexGPT-5-Lite-8B-instruct-GGUF" >&2
    echo "  C — hf.co/ai-sage/GigaChat-3.1-Lightning-10B-A1.8B-Instruct-GGUF" >&2
    exit 2
    ;;
esac

echo "→ Preset $PRESET: $DESC"
echo "→ Model:   $MODEL_TAG"
echo ""

# 1. Проверим что ollama установлен
if ! command -v ollama >/dev/null 2>&1; then
  echo "ОШИБКА: ollama CLI не найден в PATH. Установите https://ollama.com" >&2
  exit 1
fi

# 2. Пуллим модель если её ещё нет
echo "→ Проверка локального наличия модели..."
if ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -qFx "$MODEL_TAG"; then
  echo "  Модель уже загружена локально."
else
  echo "  Модель не найдена, пуллим (это может занять минуты)..."
  ollama pull "$MODEL_TAG" || {
    echo "ОШИБКА: ollama pull не прошёл. Проверьте тэг в browser:" >&2
    case "$PRESET" in
      B) echo "  https://huggingface.co/yandex/YandexGPT-5-Lite-8B-instruct-GGUF" >&2 ;;
      C) echo "  https://huggingface.co/ai-sage/GigaChat-3.1-Lightning-10B-A1.8B-Instruct-GGUF" >&2 ;;
    esac
    echo "Если квантование Q4_K_M не существует — попробуйте Q5_K_M или Q8_0 и установите MODEL_NAME явно." >&2
    exit 1
  }
fi
echo ""

# 3. Обновим .env (если файл существует, иначе создаём)
if [[ ! -f "$ENV_FILE" ]]; then
  echo "→ $ENV_FILE не существует, создаю..."
  touch "$ENV_FILE"
fi

# Удаляем старые LLM_PRESET= и закомментированные/раскомментированные MODEL_NAME=
# чтобы избежать дубликатов и конфликта override-логики.
TMP="$(mktemp)"
grep -vE '^[[:space:]]*(#[[:space:]]*)?LLM_PRESET=' "$ENV_FILE" \
  | grep -vE '^[[:space:]]*MODEL_NAME=' > "$TMP" || true
mv "$TMP" "$ENV_FILE"

echo "LLM_PRESET=$PRESET" >> "$ENV_FILE"
echo "→ $ENV_FILE: установлено LLM_PRESET=$PRESET (override MODEL_NAME удалён)"
echo ""

echo "ГОТОВО. Чтобы применить — перезапустите сервер:"
echo "  sudo systemctl restart ai-suggester"
echo ""
echo "После рестарта проверка:"
echo "  curl -s http://localhost:8000/metrics | python3 -m json.tool | grep -E '\"(model|llm_preset)\"'"
