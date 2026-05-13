# LanguageTool-RU — установка и интеграция

v2.0-b добавляет параллельный детектор стилистических и типографских правок
поверх T-lite/MorphDetector. Реализован как HTTP-клиент к локальному
LanguageTool-серверу (Java).

> **Безопасность.** Никогда не указывайте `LANGUAGETOOL_URL` на
> публичный `api.languagetool.org` — это SaaS Premium-сервиса, текст
> уходит на их инфраструктуру и не GDPR-safe. Только self-hosted локальный
> сервер.

## 1. Что даёт LanguageTool

930+ rule-based проверок для русского
([список](https://community.languagetool.org/rule/list?lang=ru)).
Категории, полезные для нашего пайплайна:

| Категория | Пример правки | Включена по умолчанию? |
|---|---|---|
| `STYLE` | «осуществлять» → «делать» (канцеляризм) | да |
| `TYPOGRAPHY` | `-` → `—` (тире), `"…"` → `«…»` | да |
| `REDUNDANCY` | «более лучший» → «лучший» | нет (можно добавить) |
| `PUNCTUATION` | пропущенная запятая | нет (T-lite ловит) |
| `GRAMMAR` | согласование, управление | нет (T-lite + MorphDetector ловят) |
| `TYPOS` | опечатки | нет (T-lite ловит) |

Включить дополнительные категории — через `LANGUAGETOOL_ENABLED_CATEGORIES`
в `.env`. Отключить — `LANGUAGETOOL_DISABLED_CATEGORIES` или
`LANGUAGETOOL_DISABLED_RULES`.

## 2. Установка LT-сервера

### 2.1 Docker (рекомендуется)

```bash
# Раз и навсегда
docker run -d --name lt --restart unless-stopped \
  -p 8081:8010 erikvl87/languagetool:6.4

# Проверка
curl http://localhost:8081/v2/languages | python3 -m json.tool | head -20
# Должны быть Russian в списке
```

Образ ~750 МБ, при старте съедает ~1 ГБ RAM. Cold-start LT-сервера
~15-30 с — после этого latency ответа /v2/check на типовой текст 700
символов ~50-200 мс.

### 2.2 Native Java (без Docker)

```bash
# Java 17+ требуется
sudo apt install openjdk-17-jre-headless

# Скачать дистрибутив (~250 МБ)
wget https://languagetool.org/download/LanguageTool-6.4.zip
unzip LanguageTool-6.4.zip
cd LanguageTool-6.4

# Запуск в HTTP-режиме
java -cp languagetool-server.jar org.languagetool.server.HTTPServer \
  --port 8081 --public --allow-origin '*'
```

Прописать systemd unit для автостарта — см. примеры в
[LT documentation](https://dev.languagetool.org/http-server).

### 2.3 Через apt (Debian/Ubuntu)

```bash
sudo apt install languagetool
# Системный пакет ставит только CLI, для HTTP-сервера используйте 2.1 или 2.2.
```

## 3. Конфигурация MainCheck

`.env`:
```bash
LANGUAGETOOL_ENABLED=true
LANGUAGETOOL_URL=http://localhost:8081
LANGUAGETOOL_LANGUAGE=ru-RU
LANGUAGETOOL_ENABLED_CATEGORIES=STYLE,TYPOGRAPHY
# Опционально:
LANGUAGETOOL_DISABLED_RULES=UPPERCASE_SENTENCE_START,WHITESPACE_RULE
LANGUAGETOOL_TIMEOUT=10
```

Перезапуск:
```bash
sudo systemctl restart ai-suggester
curl -s http://localhost:8000/metrics | python3 -m json.tool | grep languagetool
# должно: "languagetool_enabled": true, "languagetool_available": true
```

## 4. Диагностика

| Симптом | Решение |
|---|---|
| `languagetool_available: false` после restart | LT-сервер не запущен. `docker ps | grep lt` или `curl http://localhost:8081/v2/languages`. |
| LT timeout в логах ai-suggester | Поднимите `LANGUAGETOOL_TIMEOUT=20`. Или проверьте загрузку LT-сервера. |
| FP-правки в проде | Добавьте rule.id в `LANGUAGETOOL_DISABLED_RULES`. ID правила видно в `/v2/check` ответе. |
| LT не находит ошибок | Проверьте `LANGUAGETOOL_LANGUAGE=ru-RU` (не `ru`). Уточните что текст не пустой. |

## 5. Откат

```bash
# В .env
LANGUAGETOOL_ENABLED=false
sudo systemctl restart ai-suggester
# LT-сервер можно оставить запущенным или остановить:
docker stop lt
```

После отката pipeline работает идентично v1.9 + v2.0-a — LT просто не
вызывается. Тесты в репозитории покрывают этот fallback.
