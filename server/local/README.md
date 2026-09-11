# AI LibreOffice Suggester — local v9

Локальный корректор официально-делового русского текста, применяющий
**минимальные** правки. Это не сервис «улучшения текста»: любая правка
обязана быть объективной ошибкой и иметь проверяемое обоснование.

## Модель исполнения

Один systemd-сервис, один процесс Uvicorn, никаких вторых inference-серверов:

```text
.env
  ↓
ai-suggester.service   (AllowedCPUs=0-15, NUMA node0)
  ↓
uvicorn decision_app:app --host 0.0.0.0 --port 8000
  ↓
ollama.service         (AllowedCPUs=16-31, NUMA node1)
```

## Конвейер

```text
выделенный фрагмент
  ↓
segmentation            предложения + абсолютные смещения
  ↓
LocalRuleEngine         согласование (унификация признаков ИГ),
                        управление количественных слов, год,
                        согласование сказуемого с подлежащим
  ↓
DictionarySpellChecker  опечатки, доказуемые словарём OpenCorpora
  ↓
LanguageTool            пунктуация и типографика (офлайн, опционально)
  ↓
SAGE + Qwen3.5-GEC      параллельно, по предложениям
  ↓
rescue T-lite/GigaChat  только если ничего не подтверждено
  ↓
CandidateArbiter        слияние и голосование между источниками
  ↓
GenerativeGuard         классовый фильтр галлюцинаций
  ↓
DecisionEngine          адресация по смещению, защищённые термины, лимиты
  ↓
минимальные точные правки
```

## Стеки

| Stack | Rescue | Ollama | Когда использовать |
|---|---|---|---|
| **A** | T-lite-it-2.1 (8B dense) | требуется | production |
| **B** | GigaChat3.1-10B-A1.8B (MoE) | требуется | быстрее A на CPU, кандидат в production |
| **X** | нет | не требуется | минимальная latency, деградация без Ollama |
| **Y** | T-lite + GigaChat | требуется | максимальная полнота, отладка recall |

## Главный принцип качества

Модель не имеет права переписать текст пользователя. Все генеративные
стадии возвращают текст, из которого сервер извлекает **токенный** diff, и
каждая правка проходит:

1. `GenerativeGuard` — отклоняет лексические подмены словарных слов,
   падежные «улучшения» уже согласованных форм, изменения чисел и дат,
   правки внутри сокращений;
2. `CandidateArbiter` — инфлективная правка от одной модели без
   подтверждения опускается ниже порога принятия;
3. `DecisionEngine` — защищённая терминология, составные термины, лимиты
   на число и размер правок, непересечение диапазонов.

Детерминированные правила проходят guard без ограничений: они опираются
на морфологическое доказательство, а не на вероятность.

## Файлы

```text
decision_app.py            FastAPI + endpoints
hybrid_editor.py           маршрутизация стадий и rescue
decision_engine.py         итоговая сборка правок
morphology.py              согласование и словоизменение
np_agreement.py            унификация признаков именной группы
local_rules.py             детерминированные правила
spellcheck.py              словарная орфография
verification.py            guard + арбитраж
segmentation.py            предложения со смещениями
llm_text.py                нормализация ответов LLM
languagetool_stage.py      локальный LanguageTool
safe_diff.py               токенный diff со смещениями
russian_quality_models.py  SAGE (батч, опциональный int8)
ollama_gec.py              Qwen3.5-GEC + few-shot
eval/                      корпус и метрики
```

## Команды

```bash
cp -n .env.v9.example .env
./install_v9_stack.sh
sudo systemctl restart ai-suggester.service

curl http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/metrics | python3 -m json.tool

PYTHONPATH=.. python -m pytest -q test_v9_agreement.py test_v9_pipeline.py
PYTHONPATH=..:. python -m eval.run_eval --verbose
```

## Диагностика

`/metrics` → `stage_calls` показывает, какие стадии реально вызывались.
Ноль у `gec` при непустом тексте означает, что специалист не работает —
именно так вела себя v8 из-за grammar gate. `guard_rejections` показывает,
по каким причинам отклонялись правки моделей.

Разбор дефектов v8 и обоснование решений — `architecture_v9.md`.
Тюнинг под сервер — `../../Инструкции/PERFORMANCE_TUNING.md`.
