# tools/build_gec_bank

Скрипт сборки расширенного GEC-банка из открытых русскоязычных корпусов.

## Что собирает

`build.py` читает один или несколько корпусов и собирает JSONL-файл с парами
`(wrong, right)` для retrieval-augmented few-shot в `/suggest`.

Текущие источники:

| Корпус | Лицензия | Размер | Покрытие |
|---|---|---|---|
| [LORuGEC](https://github.com/ReginaNasyrova/LORuGEC) | (см. репозиторий-источник) | 960 пар, 48 правил | Punctuation, Spelling, Grammar, Semantics |

`tools/build_gec_bank/sources/` лежит в `.gitignore` — каждый разработчик скачивает
исходники локально по необходимости (это сотни мегабайт открытых данных, не нужны
в репозитории).

## Сборка

```bash
# 1. Скачать LORuGEC локально (один раз)
mkdir -p tools/build_gec_bank/sources
git clone --depth 1 https://github.com/ReginaNasyrova/LORuGEC.git \
  tools/build_gec_bank/sources/LORuGEC

# 2. Собрать расширенный банк (быстрый, чистая работа с файлом)
pip install openpyxl
python3 tools/build_gec_bank/build.py
```

Вывод: `server/shared/gec_seed/gec_bank_extended.jsonl` (~639 пар, не пересекаются
с уже существующим `gec_bank.jsonl` — итоговый суммарный объём ~927 пар).

## Подключение в сервере

В `server/local/.env`:

```
GEC_BANK_FILES=../shared/gec_seed/gec_bank.jsonl,../shared/gec_seed/gec_bank_extended.jsonl
```

Сервер при старте:
- Загружает оба файла, дедуплицирует на лету (на случай ручных правок).
- Ребилдит индекс эмбеддингов в кэш (`gec_bank.index.pkl`) — fingerprint включает
  содержимое всех файлов, поэтому добавление пар автоматически инвалидирует кэш.
- Первая сборка кэша на CPU Broadwell с `nomic-embed-text` для ~927 пар занимает
  ~2 часа. Следующие старты — секунды.

## Расширение под ваш домен

Чтобы добавить ведомственные/юридические пары без правки кода:

1. Создать свой JSONL в формате:
   ```json
   {"wrong": "должностного лиц", "right": "должностных лиц",
    "rule": "Согласование числа в приложениях",
    "definition": "...", "section": "Grammar"}
   ```
2. Положить рядом, например `server/shared/gec_seed/gec_bank_admin.jsonl`.
3. Дописать путь к `GEC_BANK_FILES` через запятую.
4. Рестарт сервера → переиндексация в фоне.

Поля `rule`, `definition`, `section` опциональны (используются только для логов /
аналитики), но желательны для отладки.

## Замена / расширение источников

Если у вас есть доступ к иным корпусам (RU-Lang8, RuCoLA-derived и т.д.) —
добавьте отдельную функцию `extract_*` в `build.py` по аналогии с
`extract_lorugec`. Каждая функция должна возвращать `list[dict]` с теми же
полями, что и существующая.
