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

### E — local edit/tagger

```text
LLM_PRESET=E
```

E не генерирует новый абзац. Он использует существующий `MorphDetector` как token-level candidate/tagger и передаёт локальные `before → after` в `DecisionEngine`.

Опубликованный `ReginaNasyrova/RussianGEC_SeqTagger` содержит код обучения и inference, но не готовый checkpoint для скачивания. Поэтому E — рабочий локальный edit-based implementation, а не выдуманный wrapper над отсутствующими весами. urlисходный Russian GEC Sequence Taggerhttps://github.com/ReginaNasyrova/RussianGEC_SeqTagger

### F — Spell-Corrector-RU-4B

```text
LLM_PRESET=F
```

F лениво загружает `melsmm/Spell-Corrector-RU-4B` через Transformers. Модель предназначена для русской орфографии, пунктуации и регистра и опубликована как готовая merged-модель. urlmodel cardhttps://huggingface.co/melsmm/Spell-Corrector-RU-4B

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

## Установка экспериментальных моделей

D и F требуют дополнительные Python-пакеты:

```bash
cd /home/service/llama/server/local
source venv/bin/activate
pip install -r requirements-experimental.txt
```

Чтобы заранее скачать все HF-веса и не ждать первый запрос:

```bash
cd /home/service/llama/server/local
bash install_experimental_models.sh
```

Скрипт кэширует:

```text
Qwen/Qwen3.5-4B
synterr-nlp/bea2026-gec-adapters
melsmm/Spell-Corrector-RU-4B
```

E дополнительных моделей не требует. G использует уже установленный T-lite через Ollama.

## D/F: параметры

```text
D_BASE_MODEL=Qwen/Qwen3.5-4B
D_ADAPTER=synterr-nlp/bea2026-gec-adapters
F_MODEL=melsmm/Spell-Corrector-RU-4B
EXPERIMENTAL_DEVICE_MAP=auto
EXPERIMENTAL_DTYPE=auto
```

Для CPU можно задать:

```text
EXPERIMENTAL_DEVICE_MAP=cpu
```

D и F загружаются лениво и исполняются в отдельном worker thread, чтобы не блокировать asyncio event loop FastAPI.

## Переключение

### A

```bash
sed -i 's/^LLM_PRESET=.*/LLM_PRESET=A/' /home/service/llama/server/local/.env
sudo systemctl restart ai-suggester.service
journalctl -u ai-suggester.service -n 120 --no-pager
```

### C

```bash
sed -i 's/^LLM_PRESET=.*/LLM_PRESET=C/' /home/service/llama/server/local/.env
sudo systemctl restart ai-suggester.service
journalctl -u ai-suggester.service -n 120 --no-pager
```

### D / E / F / G

Меняется только `LLM_PRESET`:

```text
LLM_PRESET=D
LLM_PRESET=E
LLM_PRESET=F
LLM_PRESET=G
```

После этого:

```bash
sudo systemctl restart ai-suggester.service
journalctl -u ai-suggester.service -n 160 --no-pager
```

Для D/F первый запрос может быть медленнее из-за загрузки Hugging Face weights; после кэширования веса переиспользуются.

## Логи

D:

```text
Preset=D ...
Experimental[D]: loading base=Qwen/Qwen3.5-4B ...
Experimental[D]: loading adapter=synterr-nlp/bea2026-gec-adapters
Experimental[D]: model ready
```

F:

```text
Preset=F ...
Experimental[F]: loading base=melsmm/Spell-Corrector-RU-4B ...
Experimental[F]: model ready
```

E:

```text
Preset=E ... Local edit/tagger backend ... candidates=...
```

G:

```text
Preset=G ... Local edit/tagger + T-lite verifier ...
```

## Ограничения

D и F требуют `torch/transformers/peft` и заметной RAM/CPU или GPU. A/C не требуют этих дополнительных пакетов.

E/G являются рабочими edit-based стеками, но E/G не являются буквальным запуском опубликованного checkpoint `RussianGEC_SeqTagger`, потому что готового checkpoint в исходном репозитории нет.

## Порядок тестирования

```text
A → C → D → E → G → F
```

На одном и том же наборе реальных абзацев считайте precision, recall/F0.5, false positives, гиперкоррекцию, разрушение текста, число suggestions и latency.
