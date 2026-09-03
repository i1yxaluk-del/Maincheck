# AI LibreOffice Suggester — локальный GEC-сервер

Production запускается через один systemd-сервис `ai-suggester.service` и один `uvicorn decision_app:app` на :8000.

## A/B/C: переключение полного стека

Модель и дополнительные локальные этапы выбираются одной переменной в `/home/service/llama/server/local/.env`:

```text
LLM_PRESET=A
```

После изменения достаточно:

```bash
sudo systemctl restart ai-suggester.service
journalctl -u ai-suggester.service -n 100 --no-pager
```

Preset `MODEL_NAME` переопределяет старые значения `MODEL_NAME` из `.env`, поэтому не нужно вручную удалять прежнее имя модели.

### A — T-lite baseline

```text
LLM_PRESET=A
```

`T-lite-it-2.1:q4_K_M` — текущий production baseline, который подтвердил себя на реальном официально-деловом русском. Сохраняются few-shot retrieval, MorphFilter, MorphDetector и DecisionEngine.

### B — Qwen3.5-4B

```text
LLM_PRESET=B
```

`qwen3.5:4b` — экспериментальный профиль с тем же structured-output/DecisionEngine и существующими deterministic post-processing слоями. В нашем текущем тесте он показал заметно более низкий recall, поэтому остаётся сравнительным вариантом.

### C — гибрид T-lite + compact surface GEC

```text
LLM_PRESET=C
```

Основной GEC — T-lite. После него запускается `hf.co/loqira/Qwen3.5-0.8B-GEC-KAZ-RUS-ENG:Q4_0`, который используется только как surface-correction слой для пунктуации, капитализации, опечаток и орфографии. Для качества ему нужны штатный русский system prompt, thinking OFF, temperature 0 и context 2048.

Второй слой не имеет права менять валидные словоформы, лексику или смысл. Правки принимаются только при сохранении лексических токенов либо при консервативной spelling-коррекции через `pymorphy3`. Действует `SECONDARY_GEC_MAX_EDITS` (по умолчанию 4), чтобы compact-модель не раздувала список suggestions.

Перед первым запуском C модель нужно один раз загрузить в локальный Ollama:

```bash
ollama pull hf.co/loqira/Qwen3.5-0.8B-GEC-KAZ-RUS-ENG:Q4_0
```

При старте preset C проверяет `/api/tags` и пишет в журнал, найдена ли secondary-модель. Сервис сам ничего не скачивает.

## Pipeline

```text
few-shot hybrid retrieval
        ↓
primary GEC + DecisionEngine
        ↓
MorphFilter / MorphDetector
        ↓
secondary surface GEC (только preset C)
        ↓
LanguageTool / Sage / другие опциональные стадии
        ↓
LibreOffice
```

`MORPH_DETECTOR_ENABLED`, `LANGUAGETOOL_ENABLED`, `SAGE_VALIDATOR_ENABLED` и остальные флаги можно по-прежнему включать/выключать из `.env` для A/B.

## Runtime

```text
WorkingDirectory=/home/service/llama/server/local
EnvironmentFile=/home/service/llama/server/local/.env
ExecStart=/home/service/llama/server/local/venv/bin/python3 -m uvicorn decision_app:app --host 0.0.0.0 --port 8000
```

Второй uvicorn не нужен.

## Следующие эксперименты

После честного прогона C имеет смысл сравнить три независимых направления: C без LanguageTool; C + только STYLE/TYPOGRAPHY от локального LanguageTool; и отдельный LoRA-эксперимент для Qwen3.5-4B. Такой порядок позволяет понять, какой слой повышает recall, а какой только увеличивает число suggestions.

## Эксперимент с GEC LoRA

В проекте сохранена конфигурационная точка для BEA/SyntErr adapters:

```text
GEC_ADAPTER_REPO=synterr-nlp/bea2026-gec-adapters
GEC_ADAPTER_SUBFOLDER=v4_qwen35_4b_lorugec
```

Это отдельный экспериментальный этап. Адаптер нужно сначала слить с базовой Qwen3.5 и отдельно конвертировать/квантизировать в GGUF перед Ollama.

## Требования к оценке

Модели сравниваем на одном и том же регрессионном наборе по категориям: орфография, пунктуация, согласование, управление, типографика, стиль; отдельно смотрим precision, recall/F0.5, false positives, число отображаемых suggestions и latency.
