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
3. На панели инструментов появилась **одна кнопка** «AI: Улучшить текст» — именно это увидит сотрудник.
4. Для диагностики (только у админа!) открыть **Сервис → Макросы → Мои макросы и диалоги
   → My Macros → ai_macro → Health → AICheckServer → Запустить**. Должно показать
   `[ 200 ]  <ваш URL>/health  → Ollama OK | Модель t-tech/T-lite-it-2.1:q4_K_M загружена`.

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
ollama pull nomic-embed-text    # нужен для RAG

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
# вписать OPENROUTER_API_KEY
pip install -r requirements.txt
# запустить под systemd аналогично локальному
```

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
