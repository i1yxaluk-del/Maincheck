# AI LibreOffice Suggester — локальный GEC-сервер

Профиль `qwen3.5:4b` теперь работает через отдельный **DecisionEngine**. Это важнее простой смены модели: LLM больше не формирует финальный документ и список правок одновременно.

## Архитектура

```text
LibreOffice
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
```

Ollama поддерживает structured outputs через `format` с JSON Schema; это позволяет получать машинно проверяемые объекты вместо свободного `CORRECTED + CHANGES`.

## Qwen3.5-4B

Официальный Ollama registry сейчас показывает `qwen3.5:4b` как Q4_K_M около 3.4 GB с native context 262,144 токена. Для CPU-профиля оставлены только 4096 токенов контекста и 512 токенов генерации — это сознательный latency/RAM trade-off.

## Как запустить

```bash
cp .env.qwen35.example .env
./start-qwen35.sh
```

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
- перекрывающиеся правки;
- слишком длинные кандидаты;
- изменения терминов из user dictionary;
- чистую нормализацию `ё/е`;
- идемпотентные изменения;
- больше `DECISION_MAX_CHANGES` правок.

После этого сервер сам строит `CORRECTED` и `CHANGES`. Старый LibreOffice protocol при этом сохраняется.

## Параметры CPU

```text
MODEL_NAME=qwen3.5:4b
NUM_THREADS=28
OLLAMA_NUM_CTX=4096
OLLAMA_NUM_PREDICT=512
OLLAMA_THINK=false
OLLAMA_TEMPERATURE=0
OLLAMA_KEEP_ALIVE=24h
```

Ollama предоставляет runtime-метрики `total_duration`, `load_duration`, `prompt_eval_count`, `eval_count` и длительности eval; следующим шагом benchmark должен выбирать оптимальное число CPU threads не теоретически, а измерением на конкретной VMware-гостевой системе.

## Важное про маленькую GEC-модель

Есть также отдельная Qwen3.5-0.8B GEC-модель для RU/KZ/EN в GGUF Q4_0 (~537 MB), рассчитанная на CPU и temperature 0; её заявленный training context — 2048 токенов. Она интересна как будущий дешёвый verifier, но в этом PR она не включена в основной путь, чтобы не добавлять второй LLM-запрос к каждому документу.

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
