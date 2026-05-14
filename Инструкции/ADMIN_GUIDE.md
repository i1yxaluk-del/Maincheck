# Руководство администратора

> Этот документ — для **администратора проекта**. Сотрудники, которым выдаётся
> готовый `Клиент/AI_Suggester.oxt`, не должны видеть и выполнять ничего из описанного ниже.
> Для них сценарий: получили .oxt по почте → установили → нажимают одну кнопку на
> панели. Точка.

---

## 1. Общая картина

```
  ┌──────────────┐        ┌──────────────────────┐       ┌───────────────────┐
  │  Админ       │──────► │  Клиент/AI_Suggester.oxt    │───► e-mail ───►  Работник │
  │  (этот гайд) │ 1 раз  │  (с вшитым URL)      │                 (LibreOffice)
  └──────────────┘        └──────────────────────┘       └───────────────────┘
         │
         │ поднимает и обслуживает
         ▼
  ┌──────────────┐
  │  AI-сервер   │  FastAPI + Ollama (t-tech/T-lite-it-2.1) или OpenRouter
  │  :8000       │  логи · /metrics · аудит SQLite · опционально RAG
  └──────────────┘
```

---

## 2. Подготовка .oxt для раздачи

### Шаг 1. Настроить адрес сервера

Один файл — одна строка. Откройте любым текстовым редактором:

```
Клиент/AI_Suggester/ai_macro/Settings.xba
```

Найдите функцию `GetServerList` и замените URL на адрес вашего корпоративного сервера:

```vbnet
Public Function GetServerList() As String
    GetServerList = "http://ai.corp.local:8000/suggest"
End Function
```

Можно указать несколько через `|` для автоматического fallback:

```vbnet
GetServerList = "http://ai-prime.corp.local:8000/suggest|http://ai-backup.corp.local:8000/suggest"
```

По желанию можно подкрутить в том же файле:
- `GetTimeout` — таймаут (сек), по умолчанию 120;
- `GetContextSize` — сколько символов контекста передавать модели, по умолчанию 2000;
- `GetUseTrackChanges` — применять как отслеживаемые изменения (True) или напрямую (False).

> **Никаких других файлов править не нужно.** Вся конфигурация сотрудников — в этом одном файле.

### Шаг 2. Пересобрать .oxt

Из корня репозитория:

```bash
python3 - <<'PY'
import zipfile, pathlib
root = pathlib.Path("Клиент/AI_Suggester")
out  = pathlib.Path("Клиент/AI_Suggester.oxt")
if out.exists(): out.unlink()
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(root).as_posix())
print(f"Готово: {out} ({out.stat().st_size} байт)")
PY
```

Или одной командой (если установлен `zip`):

```bash
cd Клиент/AI_Suggester && zip -r ../AI_Suggester.oxt . -x "*.DS_Store" && cd ../..
```

### Шаг 3. Проверить на себе

1. Установить собранный `Клиент/AI_Suggester.oxt` в свой LibreOffice:
   **Сервис → Управление расширениями → Добавить → выбрать .oxt**.
2. Перезапустить LibreOffice.
3. На панели инструментов появились **две кнопки**:
   - «AI: Улучшить текст» — основной макрос проверки текста.
   - «AI: В словарь» — добавить выделенный термин в пользовательский словарь сервера
     (см. [USER_DICT_GUIDE.md](USER_DICT_GUIDE.md), v1.8b).
4. Для диагностики (только у админа!) открыть **Сервис → Макросы → Мои макросы и диалоги
   → My Macros → ai_macro → Health → AICheckServer → Запустить**. Должно показать
   `[ 200 ]  <ваш URL>/health  → Ollama OK | Модель t-tech/T-lite-it-2.1:q4_K_M загружена`.
5. Управление словарём (только у админа): **Сервис → Макросы → Мои макросы
   → ai_macro → Dict** — макросы `AIDictListWords` (показать весь словарь) и
   `AIDictRemoveWord` (удалить слово).

### Шаг 4. Раздать работникам

Прикрепить `Клиент/AI_Suggester.oxt` к письму с короткой инструкцией для сотрудника
(см. `Инструкции/USER_GUIDE.md`).

---

## 3. Развёртывание сервера

### Вариант А: локальный (рекомендуется)

На сервере/workstation (T-lite-it-2.1 требует 8+ ГБ RAM и 8+ ядер; для qwen2.5:14b — 16 ГБ и 16+ ядер):

```bash
# Ollama + модель (~5 ГБ)
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama
ollama pull t-tech/T-lite-it-2.1:q4_K_M
ollama pull bge-m3              # дефолтный эмбеддер для GEC-банка (~600 МБ, v1.6.6+)
ollama pull nomic-embed-text    # fallback / эмбеддер для RAG-документов

# AI Suggester (из корня репо)
cd server/local
cp .env.example .env
# отредактировать .env при необходимости (NUM_THREADS и т.д.)
pip install -r requirements.txt
sudo cp ai-suggester.service /etc/systemd/system/
# отредактировать /etc/systemd/system/ai-suggester.service: YOUR_USERNAME, путь
sudo systemctl daemon-reload
sudo systemctl enable --now ai-suggester
```

Проверка:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

Подробности и выбор модели → [`LOCAL_MODEL.md`](LOCAL_MODEL.md).

### Вариант Б: облачный (OpenRouter)

Если у организации нет сервера с 32 ГБ RAM, но есть доступ в интернет:

```bash
cd server/cloud
cp .env.example .env
# вписать OPENROUTER_API_KEY и (опц.) CLOUD_PRESET=A|B|C|D
pip install -r requirements.txt
# запустить под systemd аналогично локальному
```

**CLOUD_PRESET** (`v2.2`):
- **A** — `openrouter/free` auto-router (дефолт, самый надёжный).
- **B** — `qwen/qwen3-next-80b-a3b-instruct:free` (рекомендуется для
  русскоязычных официальных документов).
- **C** — `google/gemma-4-31b-it:free` (multilingual, контекст 256K).
- **D** — `nvidia/nemotron-3-super-120b-a12b:free` (сильный reasoning).

Полный список моделей с fallback-цепочками для каждого preset —
в коде: `server/shared/openrouter_client.py`, словарь `CLOUD_PRESETS`.
Чтобы:
- **поменять preset «навсегда» в коде** — отредактируйте `CLOUD_PRESETS`
  и перезапустите cloud-сервер.
- **поменять модели для конкретной установки без правки кода** —
  пропишите в `server/cloud/.env`:
  ```
  OPENROUTER_MODELS=модель1,модель2,модель3
  ```
  CSV перебивает выбранный `CLOUD_PRESET`, модели пробуются по очереди.

**Архитектура cloud (`v2.2`)** — сознательно упрощён по сравнению с
локальным сервером. Сетевая модель сама справляется с орфографией,
пунктуацией, согласованием, стилем и логикой, поэтому local-specific
костыли (морф-фильтр, морф-детектор, sage, LanguageTool, few-shot
retrieval) в cloud НЕ используются и в `.env.example` отсутствуют.
Активны только:
- **RAG** (`RAG_ENABLED=true`) — фрагменты НПА РФ из `data/rag_store/`,
  чтобы модель могла ссылаться на ГОСТ Р 7.0.97-2016, методические
  рекомендации Минюста, ведомственные регламенты.
- **Пользовательский словарь** (`USER_DICT_ENABLED=true`) — REST API
  `/dict/list`, `/dict/add`, `/dict/remove`. Модель не «исправляет»
  термины из словаря.
- **Сильный SYSTEM_PROMPT** — покрывает все классы ошибок русского
  языка плюс нормы служебной переписки в РФ. См.
  `server/cloud/main.py:SYSTEM_PROMPT`.

> Cloud и local не работают вместе. Cloud сейчас — для опробования
> функционала и качества сетевой модели на реальных документах
> ведомства.

---

## 4. RAG по ведомственным документам

Если нужно научить модель понимать нормы из Гарант/КонсультантПлюс —
развёрнутое руководство → [`RAG_GUIDE.md`](RAG_GUIDE.md). Кратко:

```bash
# Один раз
ollama pull nomic-embed-text
# Положить документы в data/docs/
PYTHONPATH=server python -m shared.rag_cli ingest-folder ./data/docs

# В server/local/.env
RAG_ENABLED=true
# Перезапустить сервер
systemctl restart ai-suggester
```

Обновить редакцию:
```bash
PYTHONPATH=server python -m shared.rag_cli add data/docs/fz_44_v2025.docx --doc-id fz-44 --version 2025-03
```

Удалить отменённый документ:
```bash
PYTHONPATH=server python -m shared.rag_cli remove fz-44
```

---

## 5. Мониторинг

Все метрики лежат в одном месте:

| Интерфейс                                 | Что смотреть                         |
|-------------------------------------------|--------------------------------------|
| `curl http://ai-gw:8000/health`           | жив ли сервер и модель                |
| `curl http://ai-gw:8000/metrics?hours=24` | число/длительность запросов           |
| `tail -f logs/ai_suggester.local.log`     | живой лог                              |
| `sqlite3 logs/audit.sqlite "…"`           | кто спрашивал (запросы к `audit`)     |

Настройки retention, ротации, редакции текста — в [`LOGGING.md`](LOGGING.md).

---

## 6. Обновление расширения у сотрудников

1. Внести правки в `Клиент/AI_Suggester/ai_macro/*.xba` (например, новый URL или настройки).
2. В `Клиент/AI_Suggester/description.xml` повысить `version` (1.4.0 → 1.4.1).
3. Пересобрать `.oxt` (Шаг 2 выше).
4. Разослать по почте с инструкцией: «В LibreOffice: Сервис → Управление расширениями →
   Удалить старое AI Suggester → Добавить новое → Перезапустить».

Готовый к копипасту текст письма для сотрудников → [`USER_GUIDE.md`](USER_GUIDE.md).

---

## 7. Диагностика на стороне сервера

Типовые проблемы и их решение → [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) (раздел «Серверы»).

Быстрая проверка связности клиент ↔ сервер **без** LibreOffice:
```bash
curl -F "text=<(echo 'согласно распоряжения №5')" \
     -F "context=<(echo '')" \
     http://ai-gw:8000/suggest
```
Должен вернуть структурированный ответ с блоками `===CORRECTED=== / ===CHANGES=== / ===END===`.

---

## 8. v1.8c: sage-95m post-валидатор (опционально, default OFF)

Sage — компактная (95M параметров) русская GEC-модель `ai-forever/sage-fredt5-distilled-95m`,
которая работает как **«второе мнение»** поверх T-lite. Для каждой правки, предложенной T-lite,
sage решает «согласен / не согласен», и сомнительные правки фильтруются.

### ⚠ Важно: ограничения sage

В апреле 2026 sage уже тестировалась как **primary corrector** (см. `ЖУРНАЛ_v1.6.md`, секция
«Контекст»):

- 0/3 запятых нашла на структурной пунктуации
- стабильно галлюцинировала числа («2025» → «2015», «ТопСервис» → «Топ-Сервис»)
- обучена на типо/орфо-ошибках из Wikipedia, не на ведомственных документах

Поэтому в v1.8c sage используется **строго узко**:
- ТОЛЬКО как post-validator (фильтр правок T-lite, не как сам корректор)
- ТОЛЬКО для категории «орфография» (по default, см. `SAGE_VALIDATOR_CATEGORIES`)
- ТОЛЬКО в DRY-RUN режиме по default (только логирует, ничего не дропает)

### Рекомендованный rollout

1. **Включить `ENABLED=true`, `MODE=dryrun`** (default) на 1 неделю — sage отработает,
   и каждый verdict уйдёт в `journalctl`. Сервер по-прежнему отдаёт правки T-lite как раньше.
2. **Проанализировать логи**:
   ```bash
   sudo journalctl -u ai-suggester --since "1 week ago" | grep "Sage\["
   # Считаем: сколько DISAGREE, на каких категориях,
   # совпадают ли DISAGREE с реальными FP T-lite.
   ```
3. **Если sage уверенно ловит FP** (и редко даёт ложные DISAGREE на TP):
   `SAGE_VALIDATOR_MODE=enforce` + restart.
4. **Если sage даёт много ложных DISAGREE**: оставить в `dryrun` или выключить
   (`SAGE_VALIDATOR_ENABLED=false`).

### Установка зависимостей

```bash
cd /path/to/MainCheck
source .venv/bin/activate
pip install transformers torch sentencepiece huggingface_hub
# При первом старте сервер загрузит модель с HuggingFace (~190 МБ на диск)
```

### Включение в `.env`

```env
SAGE_VALIDATOR_ENABLED=true
SAGE_VALIDATOR_MODE=dryrun        # dryrun (default) | enforce
SAGE_VALIDATOR_DOMAIN=admin       # admin (default) | general
SAGE_VALIDATOR_CATEGORIES=орфограф  # default; "" = все категории
SAGE_VALIDATOR_DEVICE=cpu         # или cuda, если есть GPU
SAGE_VALIDATOR_WARMUP=true        # один forward pass на старте
```

Перезапустить сервер:
```bash
sudo systemctl restart ai-suggester
```

### Параметры

#### `SAGE_VALIDATOR_MODE`

- **dryrun** (default) — sage оценивает каждую правку и пишет verdict в журнал,
  **но ничего не дропает**. Безопасно: пайплайн отдаёт правки T-lite как до v1.8c.
- **enforce** — реально дропает правки по verdict + domain + categories. **Только после
  анализа dryrun-логов!**

#### `SAGE_VALIDATOR_DOMAIN` (применяется в enforce-режиме)

- **admin** (default) — приоритет **recall**. Sage дропает правку **только** если явно
  «не согласен» (verdict=DISAGREE, sage оставил `before` в своём варианте). UNKNOWN
  (sage сделал что-то третье) — правку T-lite оставляем. Рекомендуется для документов
  ведомства, КС-2, актов и т.п., где лучше показать сомнительную правку, чем пропустить
  настоящую ошибку.
- **general** — балансированно. Дропает и DISAGREE, и UNKNOWN. Для произвольных текстов,
  где precision важнее recall.

#### `SAGE_VALIDATOR_CATEGORIES` (применяется в enforce-режиме)

Substring-фильтр по тексту правки после `|`. Default: `орфограф` — только орфографические
правки могут быть дропнуты. Это страховка: sage обучена на орфографии (RUSpellRU), для
согласования/управления/пунктуации её мнение ненадёжно.

Примеры:
- `SAGE_VALIDATOR_CATEGORIES=орфограф` — только орфография (default).
- `SAGE_VALIDATOR_CATEGORIES=орфограф,пунктуация` — орфография и пунктуация.
- `SAGE_VALIDATOR_CATEGORIES=` (пустая) — все категории (НЕ рекомендуется).

### Проверка работы

```bash
curl -s http://localhost:8000/metrics | python3 -m json.tool
# Ожидаем:
#   "sage_validator_enabled": true,
#   "sage_validator_available": true,
#   "sage_validator_domain": "admin",
#   "sage_validator_model": "ai-forever/sage-fredt5-distilled-95m"
```

В логах после `/suggest`:
```
Sage[dryrun/admin/cat='согласование']: verdict=disagree для 'мероприятия'→'мероприятие'
Sage[enforce]: ДРОП правки 'опечтка'→'опечатка' (verdict=disagree, cat='орфография')
```

### Откат

`SAGE_VALIDATOR_ENABLED=false` в `.env` + `systemctl restart ai-suggester`.
Модель остаётся скачанной на диске (`~/.cache/huggingface/`), но в RAM не загружается.

Подробности архитектуры → код [`server/shared/sage_validator.py`](../server/shared/sage_validator.py).
