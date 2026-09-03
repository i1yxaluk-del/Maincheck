# AI LibreOffice Suggester — локальный GEC-сервер

Профиль `qwen3.5:4b` работает через существующий systemd-сервис `ai-suggester.service` и отдельный **DecisionEngine**. Мы не запускаем второй uvicorn поверх production-инстанса.

## Production-архитектура

```text
systemd ai-suggester.service
   ↓
uvicorn decision_app:app :8000
   ↓
существующий main.app / /suggest
   ↓
retrieval / few-shot
   ↓
Qwen3.5-4B
   ↓  structured JSON edits
DecisionEngine
   ├─ confidence threshold
   ├─ exact BEFORE validation
   ├─ overlap protection
   ├─ user-dictionary protection
   └─ ё/е normalization rejection
   ↓
server-generated CORRECTED + CHANGES
   ↓
existing deterministic post-processing
   ↓
LibreOffice
```

`decision_app:app` импортирует существующий `main.app` и подменяет только `call_ollama()`. Поэтому сохраняются существующие `/suggest`, `/health`, `/metrics`, `/dict/*`, аудит, retrieval и исторический LibreOffice protocol.

Ollama поддерживает structured outputs через `format` с JSON Schema; это позволяет получать машинно проверяемые объекты вместо свободного `CORRECTED + CHANGES`.

## Qwen3.5-4B

Официальный Ollama registry сейчас показывает `qwen3.5:4b` как Q4_K_M около 3.4 GB с native context 262,144 токена. Для CPU-профиля оставлены 4096 токенов контекста и 512 токенов генерации — это сознательный latency/RAM trade-off.

## Production через systemd

Рабочий сервер использует `EnvironmentFile`:

```text
WorkingDirectory=/home/service/llama/server/local
EnvironmentFile=/home/service/llama/server/local/.env
```

Для Qwen-профиля `ExecStart` должен указывать на тот же единственный uvicorn-процесс, но на wrapper-модуль:

```ini
ExecStart=/home/service/llama/server/local/venv/bin/python3 -m uvicorn decision_app:app --host 0.0.0.0 --port 8000
```

Команды управления остаются системными:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ai-suggester.service
sudo systemctl status ai-suggester.service
journalctl -u ai-suggester.service -n 100 --no-pager
```

`start-qwen35.sh` теперь не запускает `uvicorn`: он проверяет/скачивает модель и затем делает `sudo systemctl restart ai-suggester.service`. Это исключает конфликт за порт 8000.

## Конфигурация `.env`

```text
MODEL_NAME=qwen3.5:4b
NUM_THREADS=28
OLLAMA_NUM_CTX=4096
OLLAMA_NUM_PREDICT=512
OLLAMA_TIMEOUT=300
OLLAMA_WARMUP=true
OLLAMA_KEEP_ALIVE=24h
OLLAMA_THINK=false
OLLAMA_TEMPERATURE=0
DECISION_ENGINE_ENABLED=true
DECISION_MIN_CONFIDENCE=0.55
DECISION_MAX_CHANGES=40
DECISION_MAX_BEFORE_CHARS=180
```

В Qwen-профиле сохранены существующие retrieval и пользовательский словарь:

```text
USE_FEW_SHOT=true
GEC_TOP_K=1
GEC_RETRIEVAL_MODE=hybrid
GEC_EMBED_MODEL=bge-m3
USER_DICT_ENABLED=true
```

`MORPH_FILTER_ENABLED=true` остаётся включён. `MORPH_DETECTOR_ENABLED=false` и `LANGUAGETOOL_ENABLED=false` в этом профиле выключены намеренно: они создают независимые правки после LLM и тем самым обходят DecisionEngine. Для A/B их можно включить отдельно.

## Почему 4B GEC

Свежий BEA 2026 benchmark SyntErr публикует LoRA-адаптеры для русского GEC. Для Qwen3.5-4B опубликованный результат на LORuGEC — M2 F0.5 75.3 для режима SyntErr → LORuGEC, против 47.6 zero-shot; для Qwen3.5-9B — 70.9 против 49.2. Это делает 4B специализированный GEC особенно интересным для CPU-сервера.

В репозитории настроена точка конфигурации:

```text
GEC_ADAPTER_REPO=synterr-nlp/bea2026-gec-adapters
GEC_ADAPTER_SUBFOLDER=v4_qwen35_4b_lorugec
```

Но `GEC_ADAPTER_ENABLED=false` по умолчанию. Это намеренно: LoRA из BEA 2026 — адаптер, а не готовая Ollama-модель; его нужно отдельно собрать/подключить через совместимый Transformers/PEFT runtime. Мы не притворяемся, что `ollama pull qwen3.5:4b` автоматически загружает этот adapter.

## DecisionEngine

LLM должна вернуть:

```json
{
  "edits": [
    {
      "before": "согласно приказа",
      "after": "согласно приказу",
      "confidence": 0.97,
      "category": "government",
      "reason": "управление"
    }
  ]
}
```

DecisionEngine отклоняет:

- изменения ниже `DECISION_MIN_CONFIDENCE`;
- `before`, которого нет в исходном тексте;
- неоднозначный `before`, который встречается больше одного раза;
- перекрывающиеся правки;
- слишком длинные кандидаты;
- изменения терминов из user dictionary;
- чистую нормализацию `ё/е`;
- идемпотентные изменения;
- больше `DECISION_MAX_CHANGES` правок.

После этого сервер сам строит `CORRECTED` и `CHANGES`. Старый LibreOffice protocol при этом сохраняется.

## A/B порядок

1. T-lite — baseline.
2. Qwen3.5-4B + DecisionEngine.
3. Qwen3.5-4B + официальный BEA/SyntErr LoRA adapter — отдельный экспериментальный runtime.
4. Qwen3.5-9B + DecisionEngine — контрольный CPU-вариант.

Сравнивать нужно `precision`, `false-positive rate`, `F0.5`, median/p95 latency и tokens/s, а не только среднее число найденных ошибок.

Источники:
- https://docs.ollama.com/capabilities/structured-outputs
- https://ollama.com/library/qwen3.5:4b
- https://huggingface.co/synterr-nlp/bea2026-gec-adapters
- https://huggingface.co/loqira/Qwen3.5-0.8B-GEC-KAZ-RUS-ENG
- https://docs.ollama.com/api/usage
