# AI LibreOffice Suggester

Расширение LibreOffice Writer для осторожной коррекции официально-делового русского текста. Пользователь выделяет фрагмент, получает список локальных правок и применяет их через штатный механизм Track Changes.

## Архитектура v2

Локальный production работает в **одном** systemd-сервисе и **одном** Uvicorn-процессе:

```text
LibreOffice extension
        ↓ HTTP
server/local/decision_app.py
        ↓
       A — T-lite (production)
       F — Spell-Corrector-RU-4B (experimental)
       G — MorphDetector + T-lite verifier (experimental)
        ↓
   DecisionEngine
        ↓
  exact local edits
```

### Поддерживаемые стеки

| Stack | Назначение | Режим |
|---|---|---|
| **A** | T-lite-it-2.1 + structured edit JSON + deterministic gates | production |
| **G** | MorphDetector candidates + T-lite verifier | experimental |
| **F** | Spell-Corrector-RU-4B + morphology-preserving surface gate | experimental |

D/C/E из старых версий удалены из runtime: D не соответствует практическому latency/adapter safety на текущем сервере, C добавлял второй генерационный hop, E не был запуском готового официального checkpoint.

## Главный принцип качества

LLM не имеет права напрямую переписывать пользовательский текст в production. Стек A просит только точечные `before → after` правки в строгом JSON. `DecisionEngine` затем проверяет уверенность, точное вхождение исходного фрагмента, защищённые термины, пересечения правок и лимиты изменений.

F — исключение только для эксперимента: модель возвращает полный текст, после чего сервер извлекает локальные diff-кандидаты и пропускает их через морфологический gate. Поэтому изменения типа `изучена → изучено`, `должностного → должностных`, `деятельностей → деятельности` блокируются.

## Структура

```text
server/local/
├── decision_app.py          FastAPI + endpoints + one runtime process
├── decision_engine.py       final safety merger
├── pipelines.py             A/F/G stack implementations
├── requirements.txt         common runtime
├── requirements-experimental.txt  optional F runtime
└── test_pipelines.py        regression tests

server/shared/
├── audit.py                 SQLite request audit
├── morph_detector.py        deterministic Russian error detector
├── user_dict.py             protected terminology dictionary
└── logging_setup.py         service logging
```

`server/cloud/` остаётся отдельным интернет-зависимым вариантом и не участвует в локальном production path.

## Быстрый запуск локального production

```bash
ollama pull t-tech/T-lite-it-2.1:q4_K_M
cd server/local
cp .env.presets.example .env
pip install -r requirements.txt
sudo systemctl restart ai-suggester.service
curl http://localhost:8000/health
```

Переключение только между `A`, `F`, `G`:

```bash
./scripts/switch_llm_preset.sh A
./scripts/switch_llm_preset.sh G
./scripts/switch_llm_preset.sh F
```

## Тестирование

```bash
pytest -q server/local/test_pipelines.py
```

CI компилирует локальный сервер и запускает regression suite для защитных правил.

## Клиент

Исходники LibreOffice-расширения находятся в `Клиент/AI_Suggester`. Адрес сервера и сборка `.oxt` описаны в `Инструкции/ADMIN_GUIDE.md`.

Лицензия: MIT.
