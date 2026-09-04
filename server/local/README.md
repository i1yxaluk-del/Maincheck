# AI LibreOffice Suggester — локальный GEC-сервер

Production запускается через один systemd-сервис `ai-suggester.service` и один `uvicorn decision_app:app` на :8000. Второй uvicorn не нужен.

## Пресеты

Переключение полного стека делается одной переменной в `/home/service/llama/server/local/.env`:

```text
LLM_PRESET=A
```

После изменения:

```bash
sudo systemctl restart ai-suggester.service
journalctl -u ai-suggester.service -n 100 --no-pager
```

В репозитории остаются два базовых стека A/C и добавлены четыре экспериментальных направления D-G.

### A — T-lite baseline

```text
LLM_PRESET=A
```

`t-tech/T-lite-it-2.1:q4_K_M` — текущий production baseline для русского официально-делового текста. Используются structured output, DecisionEngine, MorphFilter/MorphDetector, few-shot retrieval и существующий postprocess.

### C — T-lite + compact surface GEC

```text
LLM_PRESET=C
```

Основной GEC — T-lite. После него запускается `hf.co/loqira/Qwen3.5-0.8B-GEC-KAZ-RUS-ENG:Q4_0` как консервативный surface-слой. Secondary не имеет права переписывать лексику/словоформы целиком; применяется только через safe merge.

Установка secondary на production host:

```bash
ollama pull hf.co/loqira/Qwen3.5-0.8B-GEC-KAZ-RUS-ENG:Q4_0
```

C остаётся experimental по качеству: используйте его только после проверки на вашем регрессионном наборе.

## Новые экспериментальные стеки D-G

Важное отличие: D-G **не направляются автоматически в Ollama**. Каждый из них требует отдельного inference backend с OpenAI-compatible endpoint `/v1/chat/completions`. Это сделано специально, чтобы нельзя было случайно загрузить несовместимую sequence-tagging модель как обычную chat-модель.

### D — Qwen3.5-4B + SyntErr→LORuGEC LoRA

```text
LLM_PRESET=D
EXPERIMENTAL_GEC_URL=http://127.0.0.1:1234
EXPERIMENTAL_MODEL=Qwen3.5-4B-SyntErr-LORuGEC
```

Идея — использовать Qwen3.5-4B с русским GEC-дообучением вместо zero-shot Qwen. Публичные SyntErr/BEA 2026 adapters содержат `v4_qwen35_4b_lorugec`, для которого на LORuGEC test опубликован M2 F0.5 75.3; это LoRA adapter, а не самостоятельная GGUF-модель, поэтому сначала нужен отдельный merge/quantize или PEFT/vLLM/llama.cpp runtime.

Источник adapter: `synterr-nlp/bea2026-gec-adapters`.

### E — Russian GEC Sequence Tagger

```text
LLM_PRESET=E
EXPERIMENTAL_GEC_URL=http://127.0.0.1:8101
EXPERIMENTAL_MODEL=RussianGEC_SeqTagger
```

Это другой класс модели: sequence tagging / edit-based GEC. Модель предсказывает локальные операции над токенами, а не переписывает весь абзац. Именно такой подход интересен нам как защита от галлюцинаций и разрушения текста.

Источник кода: `ReginaNasyrova/RussianGEC_SeqTagger`.

Для D/E backend обязан вернуть JSON, совместимый с нашим `DecisionEngine`:

```json
{
  "edits": [
    {
      "before": "после ночных наряда",
      "after": "после ночных нарядов",
      "confidence": 0.96,
      "category": "agreement",
      "reason": "согласование"
    }
  ]
}
```

### F — Spell-Corrector-RU-4B

```text
LLM_PRESET=F
EXPERIMENTAL_GEC_URL=http://127.0.0.1:1234
EXPERIMENTAL_MODEL=melsmm/Spell-Corrector-RU-4B
```

Это не замена основному GEC. Модель специализирована на русской орфографии, пунктуации и регистре. Её предполагаемая роль — отдельный surface слой с тем же принципом: только локальные правки через DecisionEngine.

Модель: `melsmm/Spell-Corrector-RU-4B`.

### G — Russian GEC Tagger + T-lite verifier

```text
LLM_PRESET=G
EXPERIMENTAL_GEC_URL=http://127.0.0.1:8101
EXPERIMENTAL_MODEL=RussianGEC_SeqTagger
```

Целевая архитектура production-класса:

```text
raw text
   ↓
Russian GEC Sequence Tagger
   ↓
локальные edit candidates
   ↓
pymorphy3 / user_dict / MorphDetector
   ↓
T-lite как verifier спорных кандидатов
   ↓
DecisionEngine
   ↓
минимальный набор CHANGES
```

G — самый интересный архитектурный вариант, но на текущем коммите verifier ещё не включён автоматически: сначала необходимо поднять tagger backend и отдельно проверить качество кандидатов.

## Переменные D-G

```text
EXPERIMENTAL_GEC_URL=http://127.0.0.1:1234
EXPERIMENTAL_MODEL=
```

Ожидается OpenAI-compatible endpoint:

```text
POST /v1/chat/completions
```

Сервер должен принимать `response_format=json_schema` и возвращать `choices[0].message.content` как JSON по схеме `edits`.

## Развёртывание A/C

Production `.env`:

```bash
cd /home/service/llama/server/local
nano .env
```

Для A:

```text
LLM_PRESET=A
```

Для C:

```text
LLM_PRESET=C
```

Затем:

```bash
sudo systemctl restart ai-suggester.service
journalctl -u ai-suggester.service -n 100 --no-pager
```

Проверка должна показывать:

```text
Preset=A ... model=t-tech/T-lite-it-2.1:q4_K_M
```

или:

```text
Preset=C ... model=t-tech/T-lite-it-2.1:q4_K_M
SecondaryGEC ready: model=...
```

## Развёртывание D-F через LM Studio / llmster

LM Studio может выступать как OpenAI-compatible локальный inference backend. Для headless Linux следует использовать server/headless runtime, а не запускать второй uvicorn нашего приложения.

Общий контракт:

```text
LibreOffice
   ↓
FastAPI :8000
   ↓
experimental backend :1234
   ↓
D/F model
```

После запуска backend:

```text
EXPERIMENTAL_GEC_URL=http://127.0.0.1:1234
```

и:

```bash
sudo systemctl restart ai-suggester.service
journalctl -u ai-suggester.service -n 100 --no-pager
```

Для E/G endpoint `:8101` предполагает отдельный небольшой Python wrapper над `RussianGEC_SeqTagger`, потому что исходный репозиторий sequence-tagger не является готовым OpenAI-compatible server.

## Важное ограничение текущего коммита

D-G — это **каркас экспериментов и безопасный routing**, а не готовые production-модели. В частности, репозиторий не пытается автоматически скачать LoRA adapter, слить его с Qwen3.5, конвертировать в GGUF или поднимать sequence-tagger HTTP wrapper. Эти операции зависят от конкретного runtime и железа.

Это намеренно: A/C остаются рабочими базовыми пресетами, а D-G нельзя случайно включить без явного `EXPERIMENTAL_GEC_URL`.

## Рекомендуемый порядок испытаний

1. A — контрольный baseline.
2. D — Qwen3.5-4B + SyntErr→LORuGEC.
3. E — sequence tagger.
4. G — sequence tagger + T-lite verifier.
5. F — отдельный surface benchmark для орфографии/пунктуации.
6. C — только как дополнительный hybrid surface эксперимент.

Для сравнения используйте один и тот же набор реальных абзацев и отдельно считайте precision, recall/F0.5, false positives, гиперкоррекции, разрушение текста, число suggestions и latency.
