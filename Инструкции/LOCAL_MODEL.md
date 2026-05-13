# Локальная LLM — подробное руководство

## Рекомендованная модель (v1.5, апрель 2026)

**`t-tech/T-lite-it-2.1:q4_K_M`** — русскоязычный instruct-tune от T-Bank на
базе Qwen3-8B, выпущен в декабре 2025. Даёт лучшее соотношение «скорость /
качество исправлений» на CPU для задачи корректуры официально-делового стиля.
После прогрева отвечает за **30–50 с** на типовой фрагмент — это **в 2×
быстрее** прежнего дефолта `qwen2.5:14b` при **идентичном качестве** правок
падежного управления, согласования и пунктуации.

> Почему именно `:2.1`, а не `:1.0`? T-lite v1.0 (середина 2024) был fine-tune
> без инструкционного тюнинга — T-Bank сам писал, что он не годится для
> production. Версия 2.1 (декабрь 2025) основана на Qwen3-8B с
> дообучением под русскую официальную речь — это первая T-lite, которую
> сам T-Bank рекомендует ставить на боевой сервис.

### Боевой бенчмарк (апрель 2026, Broadwell Xeon E5-2690 v4, 2×16 cores, 31 ГБ RAM)

Типовой фрагмент делопроизводства Росгвардии с 4 ошибками падежного управления
плюс RAG-контекст (методичка + ведомственный словарь):

| Модель | Warm | RAM | Качество исправлений | Формат ===CHANGES=== | Вердикт |
|---|---|---|---|---|---|
| **`t-tech/T-lite-it-2.1:q4_K_M`** ⭐ | **44 с** | 5 ГБ | **4/4 ✓** | Чистый; 2/5 идемпотентных пунктов (фильтруются) | **ПО УМОЛЧАНИЮ** |
| `qwen2.5:14b` | 86 с | 9 ГБ | 4/4 ✓ | Чистый | запасной baseline |
| `qwen2.5:32b` | ~100 с | 19 ГБ | 4/4 ✓ | Чистый | старый флагман |
| `forzer/GigaChat3-10B-A1.8B` | 30 с | 6 ГБ | 4/4 ✓ | ❌ Дублирует текст, копирует RAG | экспериментальный |
| `qwen3:30b-a3b-instruct-2507-q4_K_M` | 119 с* | 18 ГБ | 4/4 ✓ + новый баг | ❌ Зациклилась в CHANGES | не рекомендуется |
| `qwen3:14b` | 215 с | 9 ГБ | 4/4 ✓ | Чистый | слишком медленно |

\* На NUMA-системе с `--interleave=all` (без этого падает по OOM — см. §9).
  Без cross-socket трафика ожидание 40–60 с, но протестировать это в текущем
  конфиге не удалось.

### Таблица «модель → ресурс»

| Модель                                 | RAM (Q4) | Качество (рус) | Команда                                              |
|----------------------------------------|----------|----------------|------------------------------------------------------|
| **`t-tech/T-lite-it-2.1:q4_K_M`** ⭐   | ~5 ГБ    | Отличное       | `ollama pull t-tech/T-lite-it-2.1:q4_K_M`            |
| `qwen2.5:14b`                          | ~9 ГБ    | Хорошее        | `ollama pull qwen2.5:14b`                            |
| `qwen2.5:32b`                          | ~19 ГБ   | Отличное       | `ollama pull qwen2.5:32b`                            |
| `gemma3:27b`                           | ~17 ГБ   | Хорошее        | `ollama pull gemma3:27b`                             |
| `mistral-small3.2:24b`                 | ~14 ГБ   | Хорошее        | `ollama pull mistral-small3.2:24b`                   |
| `forzer/GigaChat3-10B-A1.8B`           | ~6 ГБ    | Хорошее, но формат | `ollama pull forzer/GigaChat3-10B-A1.8B`         |
| ~~`qwen3:30b-a3b-instruct-2507`~~      | ~18 ГБ   | Вносит новые ошибки | — (не рекомендуется)                          |
| ~~`qwen3:30b-a3b` (без 2507)~~         | ~18 ГБ   | thinking-режим игнорирует `think:false` | — (не рекомендуется)      |

### v2.0-a: A/B/C presets (май 2026)

Для быстрого переключения между моделями добавлен `LLM_PRESET` env-var. См.
`server/local/.env.example`. Helper-script: `scripts/switch_llm_preset.sh A|B|C`.

| Preset | Модель | Размер (Q4) | Лицензия | Примечание |
|---|---|---|---|---|
| A ⭐ | `t-tech/T-lite-it-2.1:q4_K_M` | ~5 ГБ | Apache 2.0 (T-Bank) | Baseline, default |
| B | `hf.co/yandex/YandexGPT-5-Lite-8B-instruct-GGUF:Q4_K_M` | ~5 ГБ | Yandex (open) | F0.5=83% LORuGEC (BEA 2025) |
| C | `hf.co/ai-sage/GigaChat-3.1-Lightning-10B-A1.8B-Instruct-GGUF:Q4_K_M` | ~6 ГБ | MIT | MoE 1.8B active, март 2026 |

A/B/C-бенчмарк: `./scripts/benchmark_llm_presets.sh /path/to/text.txt`.
Скрипт переключает preset, перезапускает сервер, гоняет один и тот же
запрос, печатает сводку.

> **Совет.** Если RAM <8 ГБ — T-lite всё равно работает (5 ГБ + ОС ≈ 6.5 ГБ).
> Если RAM ≥24 ГБ и вам нужно максимальное качество на длинных юридических
> текстах — `qwen2.5:32b` остаётся хорошим выбором, но ждать придётся ~100 с
> warm против 44 с на T-lite.

---

## 1. Установка Ollama

### Windows 10/11

1. Скачать: <https://ollama.com/download/windows>.
2. Установить (без прав администратора для профиля пользователя).
3. Проверить:

   ```cmd
   ollama --version
   ```

### Astra Linux / Ubuntu / Debian

**С правами root:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama
```

**Без root (в домашнюю папку):**
```bash
mkdir -p ~/.local/bin
curl -L https://ollama.com/download/ollama-linux-amd64 -o ~/.local/bin/ollama
chmod +x ~/.local/bin/ollama
# Добавить в PATH (один раз)
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
export PATH=$HOME/.local/bin:$PATH
# Запустить в фоне с логом
nohup ~/.local/bin/ollama serve > ~/.local/bin/ollama.log 2>&1 &
```

---

## 2. Скачивание модели

```bash
ollama pull t-tech/T-lite-it-2.1:q4_K_M
```

Объём загрузки ≈ 5 ГБ. Прогресс виден в консоли. При сбое сети можно повторять команду — Ollama докачивает.

Проверить список установленных моделей:

```bash
ollama list
```

---

## 3. Подключение к AI Suggester

1. Использовать папку `server/local/` из репозитория.
2. Скопировать `.env.example` → `.env`, задать `MODEL_NAME=t-tech/T-lite-it-2.1:q4_K_M`.
3. Запустить:

   **Linux:** `cd server/local && ./start.sh`
   **Windows:** `cd server\local && start.bat`

4. Проверить <http://localhost:8000/health> — ожидаем `Ollama OK | Модель t-tech/T-lite-it-2.1:q4_K_M загружена`.
5. Админу: для диагностики — **Сервис → Макросы → Мои макросы → ai_macro → Health → AICheckServer**.

---

## 4. Переключение между локальной и облачной моделью

В модуле `ai_macro.Settings` (Инструменты → Макросы → Мои макросы → ai_macro → Settings → `GetServerList`) указан один или несколько адресов через `|`:

```basic
GetServerList = "http://localhost:8000/suggest|https://ai.example.org/suggest"
```

Макрос перебирает адреса по очереди:

- **Основной — локальный.** Если Ollama недоступен, автоматически используется облачный (OpenRouter).
- **Чтобы перейти на облачный постоянно** — поменять адреса местами или оставить только облачный.
- **Чтобы отключить облачный** — оставить только `http://localhost:8000/suggest`.

После правки сохранить модуль (`Ctrl+S` в Basic IDE) — изменения применяются мгновенно, перезапуск не нужен.

---

## 5. Полное отключение локальной модели

Если локальный сервер временно не нужен (например, работаем только через облако):

```bash
# Остановить FastAPI-сервер AI Suggester
# Ctrl+C в окне start.sh / start.bat
# или, если как systemd-сервис:
sudo systemctl stop ai-suggester

# Остановить Ollama (освободит RAM модели)
#   Windows: правый клик по значку Ollama в трее → Quit
#   Linux с systemd:
sudo systemctl stop ollama
#   Linux без systemd:
pkill -f "ollama serve"
```

Чтобы Ollama не запускалась при входе:

- Windows: Параметры → Приложения → Автозагрузка → выключить **Ollama**.
- Linux: `sudo systemctl disable ollama` (если устанавливали через systemd).

---

## 6. Удаление модели

T-lite-it-2.1 занимает ≈ 5 ГБ на диске. Чтобы освободить место:

```bash
ollama rm t-tech/T-lite-it-2.1:q4_K_M
```

Чтобы удалить Ollama целиком:
- Windows: «Параметры → Приложения → Установленные приложения → Ollama → Удалить». Вручную удалить `%userprofile%\.ollama` (там кеш моделей).
- Linux: `sudo systemctl disable --now ollama && sudo rm -f /usr/local/bin/ollama && rm -rf ~/.ollama`.

---

## 7. Настройка под конкретное железо (`.env`)

```env
OLLAMA_URL=http://localhost:11434
MODEL_NAME=t-tech/T-lite-it-2.1:q4_K_M
NUM_THREADS=28       # ядер; оставьте 3–4 для ОС
```

**Слишком медленно?** Главный ускоритель — не смена модели, а NUMA (см. §9) и
ограничение окна контекста:

```env
OLLAMA_NUM_CTX=2048      # стандарт 4096; 2048 даёт ~2× прирост на коротких текстах
OLLAMA_NUM_PREDICT=1024  # режет длинные «развёрнутые комментарии»
```

**Нужно максимум качества на длинных юридических документах?**
```env
MODEL_NAME=qwen2.5:32b
OLLAMA_NUM_CTX=4096
```

**Недостаточно RAM?** Рекомендации:
- 8 ГБ → `t-tech/T-lite-it-2.1:q4_K_M` (5 ГБ)
- 16 ГБ → `qwen2.5:14b` (9 ГБ) как запасной к T-lite
- 24 ГБ → `qwen2.5:32b` работает впритык; оставьте запас ≥ 4 ГБ для ОС
- 32 ГБ → любая из рекомендованных

**Параллельные запросы.** По умолчанию Ollama обрабатывает запросы по одному. Если нужно несколько параллельно, запустите несколько инстансов с разными портами (`OLLAMA_HOST=127.0.0.1:11435 ollama serve`) и укажите их в `SERVER_LIST` макроса.

---

## 8. Диагностика

Из LibreOffice: панель инструментов → **AI: Проверить сервер**. Показывает HTTP-код и ответ `/health` для каждого адреса.

Из терминала:
```bash
curl http://localhost:8000/health     # статус сервера AI Suggester
curl http://localhost:11434/api/tags  # список моделей в Ollama
curl http://localhost:8000/metrics    # аудит: запросов за 24 ч, средняя длительность
```

См. также `Инструкции/TROUBLESHOOTING.md`.

---

## 9. NUMA / многосокетные серверы

На серверах с двумя и более CPU-сокетами (Broadwell Xeon E5-26xx, Skylake Gold
6xxx и т.п.) cross-socket доступ к памяти замедляет инференс в 2–3×. Быстрее
всего привязать Ollama к одному сокету и аллоцировать память ТОЛЬКО из его
NUMA-ноды:

```ini
# /etc/systemd/system/ollama.service.d/numa.conf
[Service]
ExecStart=
ExecStart=/usr/bin/numactl --cpunodebind=0 --membind=0 /usr/local/bin/ollama serve
```

T-lite-it-2.1 (5 ГБ) и qwen2.5:14b (9 ГБ) свободно помещаются в одну ноду
(обычно ~15.5 ГБ на двухсокетной системе с 31 ГБ RAM). Для них этот профиль
оптимален.

**Модели крупнее 15 ГБ** (qwen2.5:32b 19 ГБ, qwen3:30b-a3b 18 ГБ, gemma3:27b
17 ГБ) на такой конфиг НЕ поместятся — Ollama упадёт с «llama runner process
has terminated». Фикс — заменить `--membind=0` на `--interleave=all`, но
тогда половина обращений идёт через cross-socket bus и скорость падает 2×.
Поэтому на текущем железе **смысла в моделях крупнее 15 ГБ нет** — они будут
медленнее 14B-dense или T-lite-8B.

Чек NUMA-конфигурации:
```bash
lscpu | grep NUMA                     # сколько нод и как разложены ядра
numactl --hardware                    # сколько RAM на ноде
free -h                               # общий объём
cat /proc/$(pgrep ollama)/numa_maps   # что уже аллоцировано на ноде
```

Если сокет один — ничего делать не нужно, systemd override не требуется.
