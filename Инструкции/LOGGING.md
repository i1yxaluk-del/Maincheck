# Логирование, аудит и мониторинг

## Куда пишутся логи

Оба сервера (`server/local`, `server/cloud` — облачный) используют общий
модуль `shared.logging_setup`, который создаёт:

- `logs/ai_suggester.local.log`  — локальный сервер
- `logs/ai_suggester.cloud.log`  — облачный сервер
- вывод в `stderr` (видно в окне `start.sh` / `start.bat`).

## Настройка

Все параметры задаются в `.env`:

| Переменная             | По умолчанию   | Описание                                               |
|------------------------|----------------|--------------------------------------------------------|
| `LOG_LEVEL`            | `INFO`         | `DEBUG` / `INFO` / `WARNING` / `ERROR`                 |
| `LOG_DIR`              | `logs`         | Куда писать файлы логов                                |
| `LOG_RETENTION_DAYS`   | `30`           | Сколько суток хранить (0 = бесконечно)                 |
| `AUDIT_ENABLED`        | `true`         | Включить запись в SQLite                               |
| `AUDIT_DB`             | `logs/audit.sqlite` | Путь к БД аудита                                  |
| `AUDIT_REDACT_TEXT`    | `false`        | Если `true`, текст запроса не сохраняется (только sha1)|

## Ротация и очистка

- **Файлы** ротируются `TimedRotatingFileHandler` каждые сутки
  (`ai_suggester.local.log` → `ai_suggester.local.log.2024-03-15`).
  Старые файлы старше `LOG_RETENTION_DAYS` удаляются автоматически стандартным
  механизмом Python.
- **Аудит SQLite** — на каждом старте сервера выполняется `audit.purge_old()`,
  удаляя записи старше `LOG_RETENTION_DAYS`. Дополнительно purge вызывается
  при первом обращении к `/metrics`.

Принудительно очистить всё:
```bash
rm -rf logs/
```

## Аудит: кто что спрашивал

Таблица `audit` в `logs/audit.sqlite`:

| Поле           | Тип     | Пример                                   |
|----------------|---------|------------------------------------------|
| `ts`           | TEXT    | `2026-04-24T10:15:03+00:00`              |
| `client_ip`    | TEXT    | `10.0.0.14`                              |
| `user_agent`   | TEXT    | `LibreOffice/7.5 (X11; Linux x86_64; GA)`|
| `server`       | TEXT    | `local` / `cloud`                        |
| `model`        | TEXT    | `t-tech/T-lite-it-2.1:q4_K_M`                            |
| `text_len`     | INTEGER | длина исходного текста в символах        |
| `context_len`  | INTEGER | длина переданного контекста              |
| `changes_count`| INTEGER | сколько правок предложила модель         |
| `duration_ms`  | INTEGER | время ответа модели                      |
| `ok`           | INTEGER | 1 если ответ структурирован, иначе 0     |
| `error`        | TEXT    | причина сбоя (если был)                  |
| `text_sha1`    | TEXT    | sha1 исходного текста (для дедупликации) |
| `text_snippet` | TEXT    | первые 200 символов текста (пустое, если `AUDIT_REDACT_TEXT=true`) |

### Примеры запросов к аудиту

```bash
sqlite3 logs/audit.sqlite
```

Топ-10 пользователей по IP за сутки:
```sql
SELECT client_ip, COUNT(*) AS n
FROM audit
WHERE ts >= datetime('now', '-1 day')
GROUP BY client_ip ORDER BY n DESC LIMIT 10;
```

Средняя длительность ответа по моделям за неделю:
```sql
SELECT model, COUNT(*) AS n, ROUND(AVG(duration_ms)) AS avg_ms
FROM audit
WHERE ts >= datetime('now', '-7 days')
GROUP BY model;
```

Ошибки за сутки:
```sql
SELECT ts, client_ip, error
FROM audit
WHERE ok = 0 AND ts >= datetime('now', '-1 day')
ORDER BY ts DESC;
```

## Быстрая диагностика через HTTP

```bash
curl http://localhost:8000/health    # статус Ollama и модели
curl http://localhost:8000/metrics   # аудит за 24 часа (JSON)
curl 'http://localhost:8000/metrics?hours=1'   # за последний час
```

Ответ `/metrics`:
```json
{
  "server": "local",
  "model": "t-tech/T-lite-it-2.1:q4_K_M",
  "rag_enabled": true,
  "rag_documents": 37,
  "audit": {
    "enabled": true,
    "window_hours": 24,
    "total": 137,
    "ok": 134,
    "fail": 3,
    "avg_duration_ms": 18231.4,
    "avg_changes": 2.31
  }
}
```

## Конфиденциальность

Если текст официальной переписки нежелательно сохранять даже в локальной БД —
поставьте `AUDIT_REDACT_TEXT=true`. В этом случае:
- в БД остаются только метрики (длина, длительность, модель, IP, sha1),
- `text_snippet` пустой,
- sha1 всё равно пишется — чтобы считать уникальные запросы и не ломать антидубль.

Полностью отключить аудит:
```env
AUDIT_ENABLED=false
```

## Интеграция с внешним мониторингом

`/metrics` выдаёт JSON, который легко скрести по cron:

```bash
*/5 * * * * curl -s http://localhost:8000/metrics | jq '.audit' >> /var/log/ai_suggester_metrics.jsonl
```

Для полноценного Prometheus-эндпоинта достаточно добавить
`prometheus-fastapi-instrumentator` в `requirements.txt` и подключить — структура
аудита уже даёт все нужные counter/histogram.
