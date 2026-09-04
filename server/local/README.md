# AI LibreOffice Suggester — локальный GEC-сервер

Production работает через один systemd-сервис `ai-suggester.service` и один `uvicorn decision_app:app` на `:8000`. Дополнительный uvicorn не нужен.

## Пресеты

Переключение полного стека:

```text
LLM_PRESET=A
```

После изменения `.env`:

```bash
sudo systemctl restart ai-suggester.service
journalctl -u ai-suggester.service -n 120 --no-pager
```

### A — T-lite baseline

```text
LLM_PRESET=A
```

`t-tech/T-lite-it-2.1:q4_K_M` через Ollama. Это текущий production baseline. Используются structured output, DecisionEngine, MorphFilter/MorphDetector, few-shot retrieval и общий postprocess.

### C — T-lite + compact surface GEC

```text
LLM_PRESET=C
```

Основной GEC — T-lite. После него идёт `hf.co/loqira/Qwen3.5-0.8B-GEC-KAZ-RUS-ENG:Q4_0` с консервативным safe-merge.

Однократно на production host:

```bash
ollama pull hf.co/loqira/Qwen3.5-0.8B-GEC-KAZ-RUS-ENG:Q4_0
```

### D — Qwen3.5-4B + SyntErr→LORuGEC LoRA

```text
LLM_PRESET=D
```

D загружает базовую `Qwen/Qwen3.5-4B` и PEFT adapter `synterr-nlp/bea2026-gec-adapters`, subfolder `v4_qwen35_4b_lorugec`, прямо внутри того же FastAPI-процесса. Отдельный inference server не нужен.

Опубликованные эксперименты для этого adapter дают 75.3 M2 F0.5 на LORuGEC test для Qwen3.5-4B в режиме SyntErr→LORuGEC. urladapter cardhttps://huggingface.co/synterr-nlp/bea2026-gec-adapters

Однократно:

```bash
cd /home/service/llama/server/local
source venv/bin/activate
pip install -r requirements-experimental.txt
```

Weights и adapter загружаются Hugging Face при первом обращении и кэшируются локально.

### E — local edit/tagger

```text
LLM_PRESET=E
```

E не генерирует новый абзац. Он использует существующий `MorphDetector` как token-level candidate/tagger и передаёт локальные `before → after` в `DecisionEngine`.

Это сделано намеренно: опубликованный `ReginaNasyrova/RussianGEC_SeqTagger` содержит код обучения и inference, но не готовый checkpoint для скачивания, поэтому выдавать его за готовую production-модель было бы неправильно. urlисходный Russian GEC Sequence Taggerhttps://github.com/ReginaNasyrova/RussianGEC_SeqTagger

E работает из текущего production Python-окружения и не требует второго сервера.

### F — Spell-Corrector-RU-4B

```text
LLM_PRESET=F
```

F лениво загружает `melsmm/Spell-Corrector-RU-4B` через Transformers. Модель предназначена для русской орфографии, пунктуации и регистра и опубликована как готовая merged-модель. urlmodel cardhttps://huggingface.co/melsmm/Spell-Corrector-RU-4B

Однократно требуется `requirements-experimental.txt`; модель скачивается Hugging Face при первом запросе.

### G — local edit/tagger + T-lite verifier

```text
LLM_PRESET=G
```

Pipeline:

```text
raw text
  ↓
MorphDetector / local edit candidates
  ↓
T-lite verifier (Ollama)
  ↓
DecisionEngine
  ↓
минимальные CHANGES
```

G использует T-lite только как проверяющий, а не как генератор полного исправленного абзаца.

## D/F: параметры загрузки

Можно задать:

```text
D_BASE_MODEL=Qwen/Qwen3.5-4B
D_ADAPTER=synterr-nlp/bea2026-gec-adapters
F_MODEL=melsmm/Spell-Corrector-RU-4B
EXPERIMENTAL_DEVICE_MAP=auto
EXPERIMENTAL_DTYPE=auto
```

`auto` позволяет Transformers подобрать устройство. Для CPU можно явно задать `EXPERIMENTAL_DEVICE_MAP=cpu`.

## Установка D/F

```bash
cd /home/service/llama/server/local
source venv/bin/activate
pip install -r requirements-experimental.txt
```

После этого переключение остаётся тем же:

```text
LLM_PRESET=D
```

или:

```text
LLM_PRESET=F
```

и затем:

```bash
sudo systemctl restart ai-suggester.service
journalctl -u ai-suggester.service -n 120 --no-pager
```

## Проверка

A/C должны показывать Ollama model warmup, а D/E/F/G — не должны запускать Ollama warmup как основную модель.

Для D ожидается лог вида:

```text
Preset=D ...
Experimental[D]: loading base=Qwen/Qwen3.5-4B ...
Experimental[D]: loading adapter=synterr-nlp/bea2026-gec-adapters
Experimental[D]: model ready
```

Для F:

```text
Preset=F ...
Experimental[F]: loading base=melsmm/Spell-Corrector-RU-4B ...
Experimental[F]: model ready
```

Для E:

```text
Preset=E ... candidates=...
```

Для G дополнительно должна быть видна работа T-lite verifier.

## Ограничения

D и F требуют `torch/transformers/peft` и заметной RAM/CPU или GPU; это нормально для экспериментальных стеков. A/C не требуют этих дополнительных пакетов.

E/G являются рабочими edit-based стеками, но E/G не являются буквальным запуском опубликованного checkpoint `RussianGEC_SeqTagger`: checkpoint автором не опубликован в исходном репозитории. Это различие оставлено явным в архитектуре.

## Рекомендуемый порядок тестирования

```text
A → C → D → E → G → F
```

Сначала сравнивайте на одном и том же наборе из ваших реальных абзацев. Основные метрики: recall реальных ошибок, precision правок, false positives, гиперкоррекция, разрушение текста, количество suggestions и latency.
