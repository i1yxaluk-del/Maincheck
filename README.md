# AI LibreOffice Suggester

Расширение LibreOffice Writer для осторожной коррекции официально-делового
русского текста. Пользователь выделяет фрагмент, получает список локальных
правок и применяет их через штатный механизм Track Changes.

Служба работает полностью офлайн: ни текст документа, ни его фрагменты
никуда не отправляются.

## Архитектура v9 — «точность прежде всего»

Главный принцип: **корректный текст неприкосновенен.** Ошибочная правка
дороже пропущенной ошибки, потому что подрывает доверие к инструменту.
Поэтому любая морфологическая омонимия трактуется в пользу исходного
текста, а генеративная модель не может применить правку без подтверждения.

```text
LibreOffice extension
        ↓ HTTP  POST /suggest
server/local/decision_app.py
        ↓
segmentation → предложения с абсолютными смещениями
        ↓
┌────────────────────┬──────────────────┬──────────────┬─────────┬────────────┐
│ LocalRuleEngine    │ DictionarySpell  │ LanguageTool │  SAGE   │ Qwen3.5-GEC│
│ морфология,        │ Checker          │ пунктуация,  │ орфо-   │ грамматика,│
│ доказуемо          │ словарь          │ типографика  │ графия  │ few-shot   │
└────────────────────┴──────────────────┴──────────────┴─────────┴────────────┘
        ↓
rescue T-lite / GigaChat — только если ничего не подтверждено
        ↓
CandidateArbiter → голосование между источниками
        ↓
GenerativeGuard → классовый фильтр галлюцинаций
        ↓
DecisionEngine → адресация по смещению, защищённые термины, лимиты
        ↓
точные правки + блок ===CHANGES===
```

Подробный разбор дефектов v8 и решений v9 — в
[`server/local/architecture_v9.md`](server/local/architecture_v9.md).

### Поддерживаемые стеки

| Stack | Назначение | Rescue | Ollama |
|---|---|---|---|
| **A** | production | T-lite-it-2.1 (8B dense) | требуется |
| **B** | production-candidate | GigaChat3.1-10B-A1.8B (MoE, быстрее на CPU) | требуется |
| **X** | быстрый локальный путь | нет | не требуется |
| **Y** | максимальная полнота | T-lite + GigaChat | требуется |

Детерминированный слой (`LocalRuleEngine` + `DictionarySpellChecker`)
работает во всех стеках и не зависит ни от сети, ни от Ollama: при
недоступности моделей служба деградирует, но продолжает находить ошибки.

## Качество

Метрика, а не декларация: `server/local/eval/` — 70 заведомо корректных
официально-деловых предложений и 20 с ошибками известных классов.
Ключевой показатель — доля испорченных корректных предложений.

| Метрика | v8 (`1ec8c62`) | v9 |
|---|---|---|
| Испорчено корректных предложений | 49 из 70 | **0 из 70** |
| Точных исправлений | 3 из 20 | **16 из 20** |
| Precision правок | 0.093 | **1.000** |
| Recall | 0.348 | **0.826** |
| F0.5 | 0.109 | **0.960** |

Замер выполнен только детерминированными стадиями, без сети и без моделей,
поэтому воспроизводим в CI:

```bash
cd server/local
PYTHONPATH=..:. python -m eval.run_eval --verbose
PYTHONPATH=..:. python -m eval.run_eval --stack full     # включая SAGE/GEC/Ollama
```

CI-гейт запрещает мерж при `clean_damaged > 0` или `error_exact < 16`.

## Структура

```text
server/local/
├── decision_app.py         FastAPI, endpoints, единый процесс
├── hybrid_editor.py        маршрутизация стадий, сегментация, rescue
├── decision_engine.py      итоговая сборка правок и защитные лимиты
├── morphology.py           согласование и словоизменение (pymorphy3)
├── np_agreement.py         унификация признаков именной группы
├── local_rules.py          детерминированные правила
├── spellcheck.py           опечатки по словарю OpenCorpora
├── verification.py         фильтр галлюцинаций + голосование
├── segmentation.py         предложения с сохранением смещений
├── llm_text.py             нормализация ответов LLM
├── languagetool_stage.py   локальный LanguageTool
├── safe_diff.py            токенный diff с абсолютными смещениями
├── russian_quality_models.py  SAGE (батч, int8)
├── ollama_gec.py           Qwen3.5-GEC в демоне Ollama
└── eval/                   корпус и метрики качества

server/shared/
├── audit.py                аудит запросов в SQLite
├── gec_bank.py             банк эталонных пар для few-shot
├── user_dict.py            защищённая терминология
├── languagetool_client.py  HTTP-клиент LanguageTool
└── logging_setup.py        журналирование службы
```

`server/cloud/` — отдельный интернет-зависимый вариант, в локальном
production-пути не участвует.

## Быстрый запуск

```bash
cd server/local
cp -n .env.v9.example .env
./install_v9_stack.sh
sudo systemctl restart ai-suggester.service
curl http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/metrics | python3 -m json.tool
```

Переключение стека — правкой `LLM_PRESET` в `.env` (`A`, `B`, `X`, `Y`) и
перезапуском службы.

Тюнинг под конкретный сервер (NUMA, потоки, квантизация) —
[`Инструкции/PERFORMANCE_TUNING.md`](Инструкции/PERFORMANCE_TUNING.md).

## Тестирование

```bash
cd server/local
PYTHONPATH=.. python -m pytest -q test_v9_agreement.py test_v9_pipeline.py
PYTHONPATH=..:. python -m eval.run_eval --max-fp 0 --min-exact 16
```

## Клиент

Исходники расширения LibreOffice — в `Клиент/AI_Suggester`. Адрес сервера
и сборка `.oxt` описаны в `Инструкции/ADMIN_GUIDE.md`.

Лицензия: MIT.
