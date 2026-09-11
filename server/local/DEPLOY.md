# Локальное развертывание

Команды выполняются на `ext` под пользователем `service`.

## 1. Обновление кода и зависимостей

```bash
cd /home/service/llama
git pull --ff-only origin main
cd server/local
./install_v9_stack.sh
```

Скрипт ставит зависимости, компилирует модули, **прогоняет оффлайн-оценку
качества** и только затем скачивает SAGE и GEC-модель. Если оценка
падает — деплой останавливается: это защита от повторения истории v8,
когда регрессия точности прожила четыре релиза.

## 2. Конфигурация

```bash
cp -n .env.v9.example .env
```

Минимальный набор для production:

```env
LLM_PRESET=A
NUM_THREADS=16
SAGE_CORRECTOR_THREADS=8
LOCAL_RESCUE_MODE=auto
GENERATIVE_GUARD_ENABLED=true
DECISION_MIN_CONFIDENCE=0.55
```

Пунктуация и типографика (опционально, локальный LanguageTool на 8081):

```env
LANGUAGETOOL_ENABLED=true
LANGUAGETOOL_URL=http://127.0.0.1:8081
LANGUAGETOOL_ENABLED_CATEGORIES=PUNCTUATION,TYPOGRAPHY
```

Публичный `api.languagetool.org` использовать запрещено.

Переменные `LOCAL_GRAMMAR_GATE`, `LOCAL_GRAMMAR_GATE_CATEGORIES` и
`LOCAL_RESCUE_MIN_GRAMMAR_CANDIDATES` в v9 удалены и больше не читаются.

## 3. Служба

```bash
sudo cp ai-suggester.service /etc/systemd/system/ai-suggester.service
sudo systemctl daemon-reload
sudo systemctl restart ai-suggester.service
curl http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/metrics | python3 -m json.tool
journalctl -u ai-suggester.service -n 100 --no-pager
```

Unit-файл закрепляет Python-процесс на NUMA-узле 0 (`AllowedCPUs=0-15`).
Ollama нужно закрепить на узле 1 — см.
`Инструкции/PERFORMANCE_TUNING.md`.

## 4. Проверка после деплоя

Строка старта в журнале должна содержать все включённые стадии:

```
Stack=A (...), generator=..., SAGE=..., GEC=..., retrieval=2135,
rules=True, spellcheck=True, languagetool=..., rescue=auto
```

Строка обработки запроса — непустые вызовы моделей:

```
suggest v9 stack=A len=189 ... stages={'rules': 1, 'spell': 1, 'sage': 1, 'gec': 3, ...}
```

Если `'gec': 0` при непустом тексте — модель не загружена или
`OLLAMA_GEC_ENABLED=false`. Строка `Grammar gate hit`, которая в v8
появлялась на каждый запрос и отключала все модели, в v9 отсутствовать
должна полностью.

## 5. Откат

```bash
git checkout <предыдущий_тег>
sudo systemctl restart ai-suggester.service
```

Детерминированные стадии можно выключить по отдельности без откатов кода:

```env
RULE_AGREEMENT_ENABLED=false
RULE_PREDICATIVE_ENABLED=false
DICT_SPELLCHECK_ENABLED=false
```
