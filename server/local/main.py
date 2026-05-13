"""
Локальный сервер AI LibreOffice Suggester.

Использует Ollama для запуска модели без интернета.
Рекомендуемая модель (v1.5, апрель 2026): t-tech/T-lite-it-2.1:q4_K_M —
русскоязычный instruct-tune от T-Bank на базе Qwen3-8B. На CPU Broadwell
даёт warm-ответ за 30–50 с, что в 2× быстрее qwen2.5:14b при идентичном
качестве исправлений падежного управления официально-делового стиля.
Альтернативы (для нестандартного железа/требований): qwen2.5:14b,
qwen2.5:32b, forzer/GigaChat3-10B-A1.8B, qwen3:30b-a3b-instruct-2507.

Новое в v1.3:
    • Логи с ротацией и retention (LOG_RETENTION_DAYS)
    • SQLite-аудит запросов (/metrics, /audit)
    • Опциональный RAG по ведомственным документам (RAG_ENABLED=true)
Новое в v1.5:
    • Переход по умолчанию на T-lite-it-2.1 (в 2× быстрее)
    • Пост-процессор ===CHANGES===: фильтрует идемпотентные пункты «X → X»
    • v1.5.11: реконструкция CHANGES из diff, когда модель сочиняет рапорт
Новое в v1.6:
    • Retrieval-augmented few-shot (USE_FEW_SHOT=true):
      на каждый запрос подмешиваются top-K похожих пар «неправильно → правильно»
      из банка (data/gec_bank.jsonl, seed из LORuGEC, 288 пар, 48 правил).
      Академически верифицированный SOTA для русского GEC
      (Sorokin & Nasyrova, BEA @ ACL 2025).
    • Точка расширения под ведомственные документы: добавьте свой JSONL в
      GEC_BANK_FILES, он будет подмешиваться в те же few-shot примеры.
      Ноль обучения, только retrieval.
"""
from __future__ import annotations

import difflib
import os
import re
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

# Подключаем shared/ к sys.path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from shared.audit import AuditStore, Timer, count_changes  # noqa: E402
from shared.logging_setup import setup_logger  # noqa: E402

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# v2.0-a: A/B/C переключение LLM-модели через preset.
# LLM_PRESET=A — T-lite-it-2.1 (8B, baseline, default)
# LLM_PRESET=B — YandexGPT-5-Lite-8B-instruct (F0.5=83% на LORuGEC, BEA 2025)
# LLM_PRESET=C — GigaChat-3.1-Lightning-10B-A1.8B (MoE, 1.8B active, MIT)
# Если MODEL_NAME явно задан в env — он перебивает preset (для кастома).
LLM_PRESETS = {
    "A": {
        "MODEL_NAME": "t-tech/T-lite-it-2.1:q4_K_M",
        "DESCRIPTION": "T-lite-it-2.1 (baseline, dense 8B Qwen3-fine-tune от T-Bank)",
    },
    "B": {
        "MODEL_NAME": "hf.co/yandex/YandexGPT-5-Lite-8B-instruct-GGUF:Q4_K_M",
        "DESCRIPTION": "YandexGPT-5-Lite-8B-instruct (F0.5=83% на LORuGEC, ACL BEA 2025)",
    },
    "C": {
        "MODEL_NAME": "hf.co/ai-sage/GigaChat-3.1-Lightning-10B-A1.8B-Instruct-GGUF:Q4_K_M",
        "DESCRIPTION": "GigaChat-3.1-Lightning-10B-A1.8B (MoE 1.8B active, MIT, ai-sage)",
    },
}
LLM_PRESET = os.getenv("LLM_PRESET", "A").strip().upper()
_default_model = LLM_PRESETS.get(LLM_PRESET, LLM_PRESETS["A"])["MODEL_NAME"]
MODEL_NAME = os.getenv("MODEL_NAME") or _default_model
NUM_THREADS = int(os.getenv("NUM_THREADS", "28"))
# Размер окна контекста (input + output в токенах). 4096 — стандарт qwen2.5,
# но если у вас короткие тексты (<2 КБ), 2048 даёт ~2× прирост скорости
# на CPU без потери качества (модель меньше тратит на init контекста).
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
# Жёсткий лимит на длину ответа в токенах. Без лимита Ollama иногда
# дописывает «развёрнутые комментарии» — режем заранее. 1024 токена
# (~3000 символов) с запасом покрывают типовой исправленный фрагмент
# плюс блок ===CHANGES===.
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "1024"))
# Таймаут одного запроса к Ollama. Должен быть БОЛЬШЕ клиентского (Settings.xba),
# чтобы клиент успевал получить осмысленную 504 вместо «нет ответа».
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))
# Прогревать модель при старте сервера (загрузить веса в RAM, чтобы первый
# запрос пользователя не ждал 30–90 с). Отключите, если стартуете много
# инстансов на одной машине и хотите экономить RAM.
OLLAMA_WARMUP = os.getenv("OLLAMA_WARMUP", "true").lower() in ("1", "true", "yes", "on")
# Broadwell/Cascade Lake CPU с 8B Q4_K_M грузится ~120-240 с на первый chat-запрос;
# 180 с почти всегда не хватает, даём запас по умолчанию.
OLLAMA_WARMUP_TIMEOUT = float(os.getenv("OLLAMA_WARMUP_TIMEOUT", "300"))
# Сколько модель остаётся в RAM после ответа. Без этого 30B-модель выгружается
# и каждый следующий запрос ждёт ~30–90 с пока она снова грузится с диска.
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
# Отключает «thinking-режим» qwen3 (Ollama ≥ 0.9). Без этого модель пишет
# многоминутный <think>…</think> перед ответом — для правки текста это лишнее.
OLLAMA_THINK = os.getenv("OLLAMA_THINK", "false").lower() in ("1", "true", "yes", "on")
# Температура сэмплинга. По умолчанию 0 — greedy decoding, полностью
# детерминирован. До v1.6.10 хардкодилось 0.1, что давало малую, но
# воспроизводимую вариативность ответа между запусками: одинаковый текст
# мог получить разные CHANGES (в v1.6.8 ablation: run1=1 пункт detailed,
# run2=3 пункта diff-reconstruction). Это маскировало регрессии в QA и
# мешало диагностике. 0 — рекомендуется для GEC, где «верный» ответ
# единственен. Поднимите до 0.1-0.3 если нужна разнообразная генерация
# (например, для exploration в исследовании).
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))

# v1.7: морфологический фильтр галлюцинированных «улучшений» падежных
# форм через pymorphy3. Закрывает класс ошибок, который не ловит ни
# ё-фильтр, ни _drop_changes_not_in_text: модель «исправляет»
# «Подразделения» → «Подразделению», хотя обе формы валидны и контекст
# не требует именно дательного. Если pymorphy3 не установлен — фильтр
# тихо отключается, остальной пайплайн работает как раньше.
# Подробности: server/shared/morph_filter.py.
MORPH_FILTER_ENABLED = os.getenv("MORPH_FILTER_ENABLED", "true").lower() in ("1", "true", "yes", "on")

# v1.8a: морфологический ДЕТЕКТОР грамматических ошибок (numeral-noun
# disagreement, adj-noun disagreement, OOV). Запускается ПОСЛЕ T-lite,
# обогащает CHANGES пунктами, которых модель не дала. По умолчанию
# включён, latency-overhead <50 мс. Если pymorphy3 не загрузился —
# тихо отключается. Подробности: server/shared/morph_detector.py.
MORPH_DETECTOR_ENABLED = os.getenv("MORPH_DETECTOR_ENABLED", "true").lower() in ("1", "true", "yes", "on")

# v1.8b: пользовательский словарь (аббревиатуры, спец-термины, имена).
# По умолчанию включён, файл хранится в data/user_dict.json (можно
# переопределить через AI_SUGGESTER_USER_DICT_PATH). Слова инжектируются
# в SYSTEM_PROMPT и в morph_detector whitelist. REST API: GET /dict/list,
# POST /dict/add, POST /dict/remove. Подробности: server/shared/user_dict.py.
USER_DICT_ENABLED = os.getenv("USER_DICT_ENABLED", "true").lower() in ("1", "true", "yes", "on")

# v2.0-b: LanguageTool-RU параллельный детектор. Стучит в локальный
# LT-сервер (Java/Docker) и добавляет в CHANGES стилистические + типо-
# графские правки, которые T-lite/MorphDetector не дают. По умолчанию
# ОТКЛЮЧЁН — нужен запущенный LT-сервер на http://localhost:8081
# (см. Инструкции/LANGUAGETOOL.md). Если сервер недоступен — graceful
# fallback, /suggest продолжает работать без LT-правок.
# Категории по умолчанию: STYLE + TYPOGRAPHY (не дублируем GRAMMAR/TYPOS).
LANGUAGETOOL_ENABLED = os.getenv("LANGUAGETOOL_ENABLED", "false").lower() in ("1", "true", "yes", "on")
LANGUAGETOOL_URL = os.getenv("LANGUAGETOOL_URL", "http://localhost:8081")
LANGUAGETOOL_LANGUAGE = os.getenv("LANGUAGETOOL_LANGUAGE", "ru-RU")
LANGUAGETOOL_TIMEOUT = float(os.getenv("LANGUAGETOOL_TIMEOUT", "10"))
LANGUAGETOOL_ENABLED_CATEGORIES = os.getenv(
    "LANGUAGETOOL_ENABLED_CATEGORIES", "STYLE,TYPOGRAPHY"
)
LANGUAGETOOL_DISABLED_CATEGORIES = os.getenv(
    "LANGUAGETOOL_DISABLED_CATEGORIES", ""
)
LANGUAGETOOL_DISABLED_RULES = os.getenv("LANGUAGETOOL_DISABLED_RULES", "")

# RAG
RAG_ENABLED = os.getenv("RAG_ENABLED", "false").lower() in ("1", "true", "yes", "on")
RAG_STORE_DIR = os.getenv("RAG_STORE_DIR", "data/rag_store")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
RAG_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")

# Few-shot retrieval (v1.6): на каждый запрос подмешиваем top-K пар
# «неправильно → правильно» из банка как примеры формата и стиля правок.
# По умолчанию отключено (0-shot), чтобы сохранить полную обратную совместимость
# с v1.5.11. Включение → USE_FEW_SHOT=true.
USE_FEW_SHOT = os.getenv("USE_FEW_SHOT", "false").lower() in ("1", "true", "yes", "on")
GEC_BANK_FILES = [
    s.strip() for s in os.getenv(
        "GEC_BANK_FILES",
        "../shared/gec_seed/gec_bank.jsonl"
        ",../shared/gec_seed/gec_bank_extended.jsonl"
        ",../shared/gec_seed/lexify_admin.jsonl",
    ).split(",")
    if s.strip()
]
GEC_TOP_K = int(os.getenv("GEC_TOP_K", "3"))
# Эмбеддер для GEC-банка. По умолчанию переиспользует RAG_EMBED_MODEL,
# но можно задать отдельно (например, RAG=nomic-embed-text для документов
# и GEC=bge-m3 для коротких пар правок).
GEC_EMBED_MODEL = os.getenv("GEC_EMBED_MODEL", RAG_EMBED_MODEL)
# Стратегия retrieval (v1.6.7):
#   hybrid — RRF dense (cosine) + sparse (BM25). Дефолт. Лучше на длинных
#            входах: dense размывает грамматический сигнал предметной
#            лексикой, BM25 ловит точные словоформы (Sorokin & Nasyrova
#            BEA 2025).
#   dense  — pure cosine, поведение v1.6.6 (откат, если hybrid вдруг хуже).
#   sparse — pure BM25, для ablation-тестов.
GEC_RETRIEVAL_MODE = os.getenv("GEC_RETRIEVAL_MODE", "hybrid").strip().lower()
if GEC_RETRIEVAL_MODE not in ("hybrid", "dense", "sparse"):
    # Не падаем, мягко даунгрейдим до hybrid.
    GEC_RETRIEVAL_MODE = "hybrid"

# v1.7: токенизатор для sparse-половины hybrid retrieval.
#   word    — старое поведение (v1.6), токены = словоформы.
#   trigram — char-trigram (recall на морфологически близких парах).
#   both    — параллельные индексы, скоры суммируются (v1.7 default).
# Дефолт `both`: word-индекс ловит точные совпадения цитат («стоимости
# выполненных работ»), а trigram-индекс — морфологические близости
# («стоимостей» ↔ «стоимости»). RAM-overhead ~5-10 МБ, latency на
# построении +30-60 мс при старте, на запросе пренебрежимо.
GEC_BM25_TOKENIZER = os.getenv("GEC_BM25_TOKENIZER", "both").strip().lower()
if GEC_BM25_TOKENIZER not in ("word", "trigram", "both"):
    GEC_BM25_TOKENIZER = "both"

logger = setup_logger("ai_suggester.local")
audit = AuditStore()

_rag_store = None
_rag_embedder = None
_gec_bank = None
_morph_filter = None  # v1.7: lazy singleton, инициализируется ниже если MORPH_FILTER_ENABLED
_morph_detector = None  # v1.8a: детектор ошибок (lazy singleton)
_user_dict = None  # v1.8b: пользовательский словарь (lazy singleton)
_sage_validator = None  # v1.8c: sage-95m пост-валидатор (lazy singleton)
_lt_client = None  # v2.0-b: LanguageTool-RU клиент (lazy singleton)

if RAG_ENABLED:
    try:
        from shared.rag_store import OllamaEmbedder, RagStore  # noqa: E402

        _rag_store = RagStore(RAG_STORE_DIR)
        _rag_embedder = OllamaEmbedder(model=RAG_EMBED_MODEL, base_url=OLLAMA_URL)
        logger.info(
            "RAG включён: store=%s, embedder=%s, docs=%d",
            RAG_STORE_DIR, RAG_EMBED_MODEL, len(_rag_store.docs),
        )
    except Exception as e:
        logger.warning("RAG не удалось инициализировать: %s", e)
        _rag_store = None

if USE_FEW_SHOT:
    try:
        from shared.gec_bank import GecBank  # noqa: E402
        from shared.rag_store import HashingEmbedder, OllamaEmbedder  # noqa: E402

        # Переиспользуем RAG-эмбеддер, только если у GEC и RAG **одинаковая
        # модель** (экономим RAM и сетевые hops). Иначе нужен отдельный
        # Ollama-эмбеддер: GEC_EMBED_MODEL может отличаться от RAG_EMBED_MODEL,
        # например, RAG=nomic-embed-text для документов + GEC=bge-m3 для пар
        # правок. Если Ollama недоступен — HashingEmbedder как резерв
        # (лексическое пересечение вместо семантического, качество ниже,
        # но не ломается).
        if _rag_embedder is not None and GEC_EMBED_MODEL == RAG_EMBED_MODEL:
            _gec_embedder = _rag_embedder
        else:
            try:
                _gec_embedder = OllamaEmbedder(model=GEC_EMBED_MODEL, base_url=OLLAMA_URL)
                # Проверка доступности: делаем probe-эмбеддинг
                _ = _gec_embedder.embed(["probe"])
            except Exception as exc:
                logger.warning(
                    "Ollama-эмбеддер (%s) недоступен (%s), переключаюсь на HashingEmbedder",
                    GEC_EMBED_MODEL, exc,
                )
                _gec_embedder = HashingEmbedder(dim=1024)
        _gec_bank = GecBank(_gec_embedder, bm25_tokenizer=GEC_BM25_TOKENIZER)
        _here = Path(__file__).resolve().parent
        _resolved_paths = [str((_here / p) if not Path(p).is_absolute() else Path(p)) for p in GEC_BANK_FILES]
        n = _gec_bank.load_jsonl(*_resolved_paths)
        if n == 0:
            logger.warning("Few-shot retrieval включён, но банк пуст — falling back на 0-shot")
            _gec_bank = None
        else:
            # Кэш индекса рядом с первым jsonl. Fingerprint включает имя
            # эмбеддера, так что смена nomic↔hashing триггерит переиндексацию.
            _cache_path = Path(_resolved_paths[0]).with_suffix(".index.pkl")
            _gec_bank.build_index(cache_path=_cache_path)
            logger.info(
                "Few-shot retrieval включён: %d пар, top_k=%d, embedder=%s, mode=%s, bm25=%s",
                n, GEC_TOP_K, _gec_embedder.name, GEC_RETRIEVAL_MODE, GEC_BM25_TOKENIZER,
            )
    except Exception as e:
        logger.warning("Few-shot retrieval не удалось инициализировать: %s", e)
        _gec_bank = None

# v1.7: ленивый singleton морф-фильтра. pymorphy3 грузит ~50 МБ
# словарей; делаем один раз при старте, переиспользуем между запросами.
# Если pymorphy3 не установлен или не загружается — `_morph_filter`
# останется без `available=True`, и весь пост-фильтр будет no-op.
if MORPH_FILTER_ENABLED:
    try:
        from shared.morph_filter import get_morph_filter  # noqa: E402

        _morph_filter = get_morph_filter()
        if _morph_filter.available:
            logger.info("MorphFilter включён: фильтр падежных «улучшений» (pymorphy3) активен")
        else:
            logger.info(
                "MorphFilter: pymorphy3 не доступен — фильтр падежных «улучшений» отключён "
                "(установите pymorphy3 + pymorphy3-dicts-ru, см. requirements.txt)"
            )
    except Exception as e:
        logger.warning("MorphFilter не удалось инициализировать: %s", e)
        _morph_filter = None

# v1.8a: морфо-детектор. Переиспользует pymorphy3 из morph_filter
# (singleton); init почти no-op если фильтр уже подгрузил словари.
if MORPH_DETECTOR_ENABLED:
    try:
        from shared.morph_detector import get_morph_detector  # noqa: E402

        _morph_detector = get_morph_detector()
        if _morph_detector.available:
            logger.info("MorphDetector включён: детектор грамматических ошибок (pymorphy3) активен")
        else:
            logger.info(
                "MorphDetector: pymorphy3 не доступен — детектор отключён "
                "(установите pymorphy3 + pymorphy3-dicts-ru, см. requirements.txt)"
            )
    except Exception as e:
        logger.warning("MorphDetector не удалось инициализировать: %s", e)
        _morph_detector = None

# v1.8b: пользовательский словарь. Загружается из data/user_dict.json
# (или AI_SUGGESTER_USER_DICT_PATH). Если файла нет — пустой словарь.
if USER_DICT_ENABLED:
    try:
        from shared.user_dict import get_user_dict  # noqa: E402

        _user_dict = get_user_dict()
        logger.info(
            "UserDict включён: загружено %d слов(а) из %s",
            len(_user_dict.list_words()), _user_dict.path,
        )
    except Exception as e:
        logger.warning("UserDict не удалось инициализировать: %s", e)
        _user_dict = None

# v1.8c: sage-95m post-валидатор. Опционально — выключен по умолчанию,
# чтобы пакет можно было ставить без `transformers`/`torch`. Если
# SAGE_VALIDATOR_ENABLED=true, sage_validator.is_available() лениво
# загрузит модель при первом обращении (или на старте, если задан warmup).
try:
    from shared.sage_validator import get_validator as _get_sage_validator  # noqa: E402

    _sage_validator = _get_sage_validator()
    if _sage_validator.config.enabled:
        # Прокидываем загрузку при старте, чтобы первый /suggest не страдал
        # на cold-start (~3-5 c). Если transformers/torch не установлены,
        # is_available() вернёт False и валидатор станет no-op'ом.
        if _sage_validator.is_available():
            logger.info(
                "SageValidator включён: model=%s, domain=%s, device=%s",
                _sage_validator.config.model_name,
                _sage_validator.config.domain,
                _sage_validator.config.device,
            )
        else:
            logger.warning(
                "SageValidator: SAGE_VALIDATOR_ENABLED=true, но модель не "
                "загрузилась — валидатор работает как no-op. Проверьте, что "
                "установлены transformers/torch и доступен HuggingFace Hub."
            )
    else:
        logger.info("SageValidator: отключён (SAGE_VALIDATOR_ENABLED=false)")
except Exception as e:  # noqa: BLE001
    logger.warning("SageValidator не удалось инициализировать: %s", e)
    _sage_validator = None

# v2.0-b: LanguageTool-RU клиент. Опционально (LANGUAGETOOL_ENABLED).
# Проверка доступности LT-сервера лениво при первом запросе, чтобы не
# падать на старте если LT-сервер ещё не поднят.
if LANGUAGETOOL_ENABLED:
    try:
        from shared.languagetool_client import (  # noqa: E402
            get_languagetool_client,
            _parse_csv_env,
        )

        _lt_client = get_languagetool_client(
            url=LANGUAGETOOL_URL,
            language=LANGUAGETOOL_LANGUAGE,
            enabled_categories=_parse_csv_env(LANGUAGETOOL_ENABLED_CATEGORIES),
            disabled_categories=_parse_csv_env(LANGUAGETOOL_DISABLED_CATEGORIES),
            disabled_rules=_parse_csv_env(LANGUAGETOOL_DISABLED_RULES),
            timeout=LANGUAGETOOL_TIMEOUT,
        )
        # Один probe-запрос при старте чтобы залогировать наличие сервера.
        # Если LT недоступен — _lt_client.available == False, /suggest
        # просто пропускает LT-этап (не падает).
        if _lt_client.available:
            logger.info(
                "LanguageTool включён: url=%s, language=%s, enabled_categories=%s",
                LANGUAGETOOL_URL, LANGUAGETOOL_LANGUAGE,
                LANGUAGETOOL_ENABLED_CATEGORIES or "(все)",
            )
        else:
            logger.warning(
                "LanguageTool: LANGUAGETOOL_ENABLED=true, но сервер %s "
                "недоступен. /suggest продолжит работу без LT-правок.",
                LANGUAGETOOL_URL,
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("LanguageTool не удалось инициализировать: %s", e)
        _lt_client = None
else:
    logger.info("LanguageTool: отключён (LANGUAGETOOL_ENABLED=false)")


SYSTEM_PROMPT = """Ты — корректор русского языка для официальных документов. Не рассуждай, сразу выдавай ответ в нужном формате.

ИСПРАВЛЯЙ ТОЛЬКО ЯВНЫЕ ОШИБКИ:
• орфография — опечатки, удвоение/пропуск букв, слитное/раздельное написание;
• управление — «согласно приказу» (не «согласно приказа»), «благодаря решению»;
• согласование — однородные члены и причастные обороты в одном роде, числе и падеже с главным словом, даже если оно стоит за несколько слов («актов, …, не предусмотренных», не «предусмотренной»);
• пунктуация — запятые при однородных членах, обособленных оборотах, придаточных.

НЕ ТРОГАЙ:
• аббревиатуры и сокращения (п/п, вх.№, исх.№, ФСБ, МВД);
• ведомственные термины и профессиональные обороты;
• правильно написанный текст («улучшать стиль» нельзя);
• е/ё взаимозаменяемы — это стилистика, не ошибка;
• структуру и смысл предложений.

ФОРМАТ ОТВЕТА (строго, без какого-либо текста до или после):
===CORRECTED===
<исправленный текст>
===CHANGES===
1. «было» → «стало» | краткая причина (5–10 слов)
===END===

ПРАВИЛА ЦИТИРОВАНИЯ В CHANGES (важно — клиент применяет правки поиском по тексту):
• в кавычках «было» цитируй ТОЧНО как в исходном тексте, побуквенно;
• БЕЗ многоточия (… или ...), БЕЗ сокращений, БЕЗ перефразирования;
• «стало» — фрагмент той же длины с применённой правкой;
• если правка касается запятой — включи в цитату слова слева и справа от запятой;
• это правило относится к ФОРМАТУ цитаты в CHANGES, а не к тому, какие ошибки искать. Ищи все ошибки правописания, управления, согласования, пунктуации одинаково внимательно.

Если ошибок нет:
===CORRECTED===
<исходный текст без изменений>
===CHANGES===
1. Ошибок не найдено. Текст соответствует нормам.
===END==="""


app = FastAPI(title="AI LibreOffice Suggester — Local", version="1.6.0")


# ─── Нормализация переносов строк (v2.2 Shift+Enter fix) ─────────────
# LibreOffice getString() возвращает paragraph-break как \r или \r\n
# (платформо-зависимо), а Shift+Enter (мягкий перенос) — как \n или
# U+2028. Конвенция, общая с client-side ApplyWholeReplace:
#   • \n\n (двойной LF) — граница абзаца;
#   • \n     (одиночный LF) — мягкий перенос внутри абзаца.
# Применяется только к raw_text/raw_ctx на входе /suggest; внутренние
# helpers (тесты, постпроцессинг) работают с уже нормализованным текстом.


def _normalize_line_breaks(text: str) -> str:
    """\\r\\n, \\r, U+2028 → консистентная \\n-конвенция.

    Без этой функции одиночный Chr(10) от Shift+Enter после round-trip
    через LLM мог склеиваться или, наоборот, превращаться в paragraph
    break при applyWholeReplace в Main.xba — клиент видел «разъединение»
    одного фрагмента на несколько абзацев. См. ЖУРНАЛ_v1.6.md v2.2.
    """
    if not text:
        return text
    text = text.replace("\r\n", "\n\n")
    text = text.replace("\r", "\n\n")
    text = text.replace("\u2028", "\n")
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text


@app.on_event("startup")
async def _warmup_ollama():
    """Грузим модель в RAM при старте сервера, чтобы первый запрос
    пользователя не ждал 30–90 с на загрузку весов 8B-модели.

    Делает один chat-запрос с **тем же `num_ctx`, что в проде** — Ollama
    при этом аллоцирует kv-cache нужного размера. Без этого первый
    реальный запрос с длинным промптом вызывал переаллокацию kv-cache
    (warmup делал 512, реальный запрос требовал 2048-4096) и cold-старт
    рос до 100 с при warm 79 с (см. v1.6.8 ablation в ЖУРНАЛ_v1.6.md).

    Дополнительно прогреваем decode-путь (`num_predict=8`), чтобы Ollama
    скомпилировала ядра BLAS и токенизатор именно под русскую раскладку,
    а не под "ok" из v1.6.x ≤ 1.6.9.
    """
    if not OLLAMA_WARMUP:
        return
    preset_desc = LLM_PRESETS.get(LLM_PRESET, {}).get(
        "DESCRIPTION", "custom (MODEL_NAME override)"
    )
    logger.info(
        "v2.0-a LLM preset=%s — %s",
        LLM_PRESET, preset_desc,
    )
    logger.info(
        "Прогрев модели %s через Ollama (num_ctx=%d, timeout=%.0fs)…",
        MODEL_NAME, OLLAMA_NUM_CTX, OLLAMA_WARMUP_TIMEOUT,
    )
    # Реалистичный по длине dummy-промпт (~600 токенов на T-lite) —
    # достаточно, чтобы kv-cache аллоцировался на полный OLLAMA_NUM_CTX,
    # но не настолько, чтобы старт занял минуту. Текст подобран
    # специально как «нечего исправлять» — модель не должна тратить
    # время на длинную генерацию правок.
    warmup_payload = (
        "Контрольный прогон системы при старте сервера. "
        "Этот текст не содержит ошибок и используется только "
        "для предварительной аллокации kv-cache в Ollama, "
        "чтобы первый пользовательский запрос не ждал переинициализации "
        "контекста. Никаких действий по этому тексту выполнять не нужно."
    ) * 6
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_WARMUP_TIMEOUT) as c:
            r = await c.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": warmup_payload + "\n\n/no_think"}],
                    "stream": False,
                    "think": False,
                    "keep_alive": OLLAMA_KEEP_ALIVE,
                    "options": {
                        "num_ctx": OLLAMA_NUM_CTX,
                        "num_predict": 8,
                        "num_thread": NUM_THREADS,
                        "temperature": OLLAMA_TEMPERATURE,
                    },
                },
            )
            r.raise_for_status()
        logger.info(
            "Прогрев OK: модель загружена, kv-cache аллоцирован на num_ctx=%d, "
            "keep_alive=%s",
            OLLAMA_NUM_CTX, OLLAMA_KEEP_ALIVE,
        )
    except Exception as e:
        logger.warning("Прогрев модели не удался (%s) — первый запрос будет медленнее", e)


# ─── Пост-обработка перенесена в server/shared/postprocess.py (v2.1) ──
# Здесь оставлены только тонкие обёртки для service-dependent функций,
# которые сохраняют исторические сигнатуры (одно- или двухаргументные).
# Pure-helpers импортируются напрямую — их имена сохраняются, тесты
# вида `local_module._drop_eyo_substitutions(...)` продолжают работать.
from shared.postprocess import (  # noqa: E402
    _CHANGE_PAIR_RE,
    _CHANGE_NUM_RE,
    _THINK_BLOCK,
    _complete_changes_from_corrected,
    _drop_changes_not_in_text,
    _drop_eyo_substitutions,
    _drop_idempotent_changes,
    _expand_word_context,
    _extract_corrected_body,
    _had_any_change_pairs,
    _has_real_change_items,
    _is_eyo_only_substitution,
    _parse_change_pair_robust,
    _rebuild_changes_from_diff,
    _renumber_changes,
    _replace_changes_block,
    _replace_corrected_body,
    _strip_thinking,
    _undo_eyo_in_corrected_block,
    _undo_eyo_in_text,
)
from shared.postprocess import (  # noqa: E402
    _drop_morph_case_substitutions as _shared_drop_morph_case_substitutions,
    _drop_user_dict_changes as _shared_drop_user_dict_changes,
    _enrich_changes_with_detector as _shared_enrich_changes_with_detector,
    _enrich_changes_with_languagetool as _shared_enrich_changes_with_languagetool,
    _filter_changes_with_sage as _shared_filter_changes_with_sage,
)


def _drop_user_dict_changes(text: str) -> str:
    """Тонкая обёртка: автоматически передаёт глобальный `_user_dict`."""
    return _shared_drop_user_dict_changes(text, _user_dict)


def _drop_morph_case_substitutions(text: str, raw_text: str) -> str:
    """Тонкая обёртка: автоматически передаёт глобальный `_morph_filter`."""
    return _shared_drop_morph_case_substitutions(text, raw_text, _morph_filter)


def _enrich_changes_with_detector(text: str, raw_text: str) -> str:
    """Тонкая обёртка: автоматически передаёт `_morph_detector` и `_user_dict`."""
    return _shared_enrich_changes_with_detector(
        text, raw_text, _morph_detector, user_dict=_user_dict,
    )


def _enrich_changes_with_languagetool(text: str, raw_text: str) -> str:
    """Тонкая обёртка: автоматически передаёт `_lt_client` и `_user_dict`."""
    return _shared_enrich_changes_with_languagetool(
        text, raw_text, _lt_client, user_dict=_user_dict,
    )


def _filter_changes_with_sage(text: str, raw_text: str) -> str:
    """Тонкая обёртка: автоматически передаёт `_sage_validator`."""
    return _shared_filter_changes_with_sage(text, raw_text, _sage_validator)



async def call_ollama(messages: list) -> str:
    # /no_think — soft-switch Qwen3, должен стоять в последнем user-сообщении
    # (не в system-prompt). Работает на любой Ollama, в т.ч. старее 0.9.
    msgs = [dict(m) for m in messages]
    if msgs and msgs[-1].get("role") == "user" and not OLLAMA_THINK:
        msgs[-1]["content"] = msgs[-1]["content"].rstrip() + "\n\n/no_think"

    payload = {
        "model": MODEL_NAME,
        "messages": msgs,
        "stream": False,
        "think": OLLAMA_THINK,  # для Ollama ≥ 0.9; старые игнорируют поле
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            # v1.6.10: дефолт 0 (greedy) — детерминированный вывод между
            # одинаковыми запросами. До v1.6.10 хардкодилось 0.1; ablation
            # v1.6.8 (run1 vs run2 на КС-2) показал, что даже 0.1 даёт
            # разные CHANGES для одного текста — это маскирует регрессии
            # и мешает диагностике.
            "temperature": OLLAMA_TEMPERATURE,
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": OLLAMA_NUM_PREDICT,
            "num_thread": NUM_THREADS,
            "repeat_penalty": 1.1,
        },
    }
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        r.raise_for_status()
        raw = r.json()["message"]["content"].strip()
        return _drop_idempotent_changes(_strip_thinking(raw))


def _rag_context(text: str) -> str:
    """Возвращает дополнительный блок с фрагментами из RAG-хранилища."""
    if not (_rag_store and _rag_embedder):
        return ""
    try:
        hits = _rag_store.search(text, top_k=RAG_TOP_K, embedder=_rag_embedder)
    except Exception as e:
        logger.warning("RAG поиск провалился: %s", e)
        return ""
    if not hits:
        return ""
    parts = ["ПРИМЕНИМЫЕ НОРМАТИВНЫЕ ФРАГМЕНТЫ (используйте как справку, не цитируйте в CHANGES):"]
    for h in hits:
        parts.append(f"— [{h['doc_id']}] {h['text'][:600]}")
    return "\n".join(parts)


@app.get("/health", response_class=PlainTextResponse)
async def health():
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{OLLAMA_URL}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
        status = "Ollama OK"
        if MODEL_NAME in models or any(MODEL_NAME.split(":")[0] in m for m in models):
            status += f" | Модель {MODEL_NAME} загружена"
        else:
            status += f" | ВНИМАНИЕ: модель {MODEL_NAME} не найдена. Загружены: {', '.join(models)}"
        if RAG_ENABLED and _rag_store:
            status += f" | RAG: {len(_rag_store.docs)} документов"
        if _gec_bank is not None:
            status += f" | Few-shot: {len(_gec_bank)} пар, top_k={GEC_TOP_K}"
        return status
    except Exception as e:
        return f"ОШИБКА: Ollama недоступна — {e}"


@app.get("/metrics")
async def metrics(hours: int = 24):
    return JSONResponse({
        "server": "local",
        "model": MODEL_NAME,
        "llm_preset": LLM_PRESET,
        "llm_preset_description": LLM_PRESETS.get(LLM_PRESET, {}).get(
            "DESCRIPTION", "custom (MODEL_NAME override)"
        ),
        "rag_enabled": RAG_ENABLED,
        "rag_documents": len(_rag_store.docs) if _rag_store else 0,
        "rag_chunks": len(_rag_store.entries) if _rag_store else 0,
        "rag_doc_ids": (
            sorted(_rag_store.docs.keys()) if _rag_store else []
        ),
        "rag_top_k": RAG_TOP_K if (RAG_ENABLED and _rag_store) else 0,
        "rag_embedder": (
            _rag_embedder.name if _rag_embedder is not None else None
        ),
        "few_shot_enabled": _gec_bank is not None,
        "few_shot_pairs": len(_gec_bank) if _gec_bank else 0,
        "few_shot_top_k": GEC_TOP_K if _gec_bank else 0,
        "few_shot_embedder": (
            _gec_bank.embedder.name if _gec_bank is not None else None
        ),
        "few_shot_retrieval_mode": (
            GEC_RETRIEVAL_MODE if _gec_bank is not None else None
        ),
        "few_shot_bm25_terms": (
            _gec_bank.stats().get("bm25_terms", 0) if _gec_bank is not None else 0
        ),
        "audit": audit.stats(hours=hours),
        "morph_detector_enabled": _morph_detector is not None and _morph_detector.available,
        "user_dict_enabled": _user_dict is not None,
        "user_dict_size": len(_user_dict.list_words()) if _user_dict is not None else 0,
        # v1.8c: статус sage-95m post-валидатора
        "sage_validator_enabled": (
            _sage_validator is not None and _sage_validator.config.enabled
        ),
        "sage_validator_available": (
            _sage_validator is not None and _sage_validator.is_available()
        ),
        "sage_validator_domain": (
            _sage_validator.config.domain if _sage_validator is not None else None
        ),
        "sage_validator_model": (
            _sage_validator.config.model_name if _sage_validator is not None else None
        ),
        # v2.0-b: статус LanguageTool-RU параллельного детектора
        "languagetool_enabled": LANGUAGETOOL_ENABLED,
        "languagetool_available": (
            _lt_client is not None and _lt_client.available
        ),
        "languagetool_url": LANGUAGETOOL_URL if LANGUAGETOOL_ENABLED else None,
        "languagetool_language": LANGUAGETOOL_LANGUAGE if LANGUAGETOOL_ENABLED else None,
        "languagetool_enabled_categories": (
            LANGUAGETOOL_ENABLED_CATEGORIES if LANGUAGETOOL_ENABLED else None
        ),
    })


# ─── v1.8b: REST endpoints для пользовательского словаря ─────────────


@app.get("/dict/list")
async def dict_list():
    """Возвращает текущий список слов пользовательского словаря.

    Response: {"words": ["ЦСН", "КС-2", ...]} (отсортированный).

    Если словарь отключён (USER_DICT_ENABLED=false) — 503.
    """
    if _user_dict is None:
        return JSONResponse(
            {"error": "пользовательский словарь отключён"}, status_code=503,
        )
    return JSONResponse({"words": _user_dict.list_words()})


@app.post("/dict/add")
async def dict_add(request: Request):
    """Добавляет слово в пользовательский словарь.

    Request body: {"word": "ЦСН"} (JSON).
    Response: {"added": true, "total": 5} либо {"added": false, ...} если уже было.
    Errors: 400 на невалидный ввод, 503 если словарь отключён.
    """
    if _user_dict is None:
        return JSONResponse(
            {"error": "пользовательский словарь отключён"}, status_code=503,
        )
    try:
        from shared.user_dict import UserDictError  # noqa: E402
        body = await request.json()
        word = body.get("word") if isinstance(body, dict) else None
        if not isinstance(word, str):
            return JSONResponse({"error": "ожидается JSON-поле 'word'"}, status_code=400)
        added = _user_dict.add(word)
        return JSONResponse({
            "added": added, "total": len(_user_dict.list_words()),
        })
    except UserDictError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        logger.exception("dict/add failed")
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


@app.post("/dict/remove")
async def dict_remove(request: Request):
    """Удаляет слово из пользовательского словаря.

    Request body: {"word": "ЦСН"} (JSON).
    Response: {"removed": true, "total": 4} либо {"removed": false, ...} если не было.
    """
    if _user_dict is None:
        return JSONResponse(
            {"error": "пользовательский словарь отключён"}, status_code=503,
        )
    try:
        from shared.user_dict import UserDictError  # noqa: E402
        body = await request.json()
        word = body.get("word") if isinstance(body, dict) else None
        if not isinstance(word, str):
            return JSONResponse({"error": "ожидается JSON-поле 'word'"}, status_code=400)
        removed = _user_dict.remove(word)
        return JSONResponse({
            "removed": removed, "total": len(_user_dict.list_words()),
        })
    except UserDictError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        logger.exception("dict/remove failed")
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


@app.post("/suggest", response_class=PlainTextResponse)
async def suggest(
    request: Request,
    text: UploadFile = File(...),
    context: UploadFile = File(...),
):
    raw_text = (await text.read()).decode("utf-8", errors="replace").strip()
    raw_ctx = (await context.read()).decode("utf-8", errors="replace").strip()
    if not raw_text:
        return "ОШИБКА: Пустой текст"

    # v2.2: нормализация переносов (Shift+Enter fix). См. _normalize_line_breaks.
    raw_text = _normalize_line_breaks(raw_text)
    raw_ctx = _normalize_line_breaks(raw_ctx)

    extra = _rag_context(raw_text)
    user_msg = f"Контекст:\n{raw_ctx}\n"
    if extra:
        user_msg += f"\n{extra}\n"
    user_msg += f"\n---\nТЕКСТ ДЛЯ ПРОВЕРКИ:\n{raw_text}"

    # Few-shot retrieval (v1.6): подмешиваем top-K примеров перед user-сообщением.
    # Академический SOTA для RU GEC — именно эта схема (Sorokin & Nasyrova 2025).
    # Если банк пуст/недоступен — falling back на 0-shot (полная совместимость).
    few_shot_examples: list = []
    if _gec_bank is not None:
        try:
            if GEC_RETRIEVAL_MODE == "sparse":
                hits = _gec_bank.search_sparse(raw_text, top_k=GEC_TOP_K)
            elif GEC_RETRIEVAL_MODE == "dense":
                hits = _gec_bank.search(raw_text, top_k=GEC_TOP_K)
            else:  # hybrid (default)
                hits = _gec_bank.search_hybrid(raw_text, top_k=GEC_TOP_K)
            few_shot_examples = [pair for score, pair in hits]
            if few_shot_examples:
                logger.info(
                    "Few-shot (%s): подмешиваю %d пар: %s",
                    GEC_RETRIEVAL_MODE,
                    len(few_shot_examples),
                    [p.rule or p.wrong[:40] for p in few_shot_examples],
                )
        except Exception as e:
            logger.warning("Few-shot retrieval провалился (0-shot fallback): %s", e)
            few_shot_examples = []

    # v1.8b: подмешиваем пользовательский словарь в SYSTEM_PROMPT.
    # Если словарь пустой — суффикс пустой, промпт совпадает с базовым.
    effective_system_prompt = SYSTEM_PROMPT
    if _user_dict is not None:
        dict_suffix = _user_dict.render_for_prompt()
        if dict_suffix:
            effective_system_prompt = SYSTEM_PROMPT + "\n\n" + dict_suffix

    if few_shot_examples:
        from shared.gec_bank import build_few_shot_messages  # noqa: E402

        messages = build_few_shot_messages(
            system_prompt=effective_system_prompt,
            user_text=user_msg,
            examples=few_shot_examples,
        )
    else:
        messages = [
            {"role": "system", "content": effective_system_prompt},
            {"role": "user", "content": user_msg},
        ]

    client_ip = request.client.host if request.client else ""
    user_agent = request.headers.get("user-agent", "")
    timer = Timer()
    ok, error, result = True, "", ""
    with timer:
        try:
            result = await call_ollama(messages)
            if "===CORRECTED===" not in result:
                result = (
                    "===CORRECTED===\n"
                    f"{result}\n"
                    "===CHANGES===\n"
                    "1. Формат ответа не распознан — проверьте вручную.\n"
                    "===END==="
                )
            # Гарантия терминатора. Некоторые модели (yandex-corrector
            # на нестандартном chat-template, hf.co/issai/Qolda_GGUF, и др.)
            # стабильно отдают ===CORRECTED===/===CHANGES===, но забывают
            # дописать ===END===. Без него клиент v1.5.7 в fallthrough-ветке
            # на не-2xx HTTP считает ответ обрезанным и показывает «Все
            # серверы недоступны». Дописываем терминатор однократно.
            if "===END===" not in result:
                result = result.rstrip() + "\n===END==="
            # Снапшот ДО фильтра: были ли вообще «X» → «Y» пары?
            # Это нужно чтобы отличить «модель отдала пары, фильтр их
            # выкинул» (реконструируем из diff) от «модель отдала garbage,
            # сервер обернул в стаб» (реконструировать нельзя).
            had_pairs_pre_filter = _had_any_change_pairs(result)
            # v1.6.8: дропаем стилистические правки ё↔е («проведенными» →
            # «проведёнными» и т.п.) и откатываем их в CORRECTED.
            # T-lite стабильно делает такие подмены, не помогает даже
            # явный запрет в SYSTEM_PROMPT (lost-in-the-middle).
            result = _drop_eyo_substitutions(result, raw_text)
            # v1.6.9: посимвольный откат ё→е в CORRECTED закрывает bypass
            # для compound-цитат, где модель упаковала ё-подмену вместе с
            # реальной правкой («повлекших риски … Подразделения» →
            # «повлёкших риски … Подразделению»). Line-level фильтр таких
            # не дропает (внутри есть НЕ-ё разница), но в CORRECTED ё всё
            # равно остаётся. Char-level выравнивание через difflib
            # восстанавливает букву по raw_text.
            result = _undo_eyo_in_corrected_block(result, raw_text)
            # v1.7: дроп галлюцинированных «улучшений» падежных форм
            # через pymorphy3. Закрывает компанионную часть compound-bypass
            # (Подразделения → Подразделению — после v1.6.9 ё откатывается,
            # но падежная подмена остаётся). Логика: одна лемма, тот же
            # number, разный case, нет case-governing предлога перед
            # `before` в raw_text → дроп. См. _drop_morph_case_substitutions
            # и server/shared/morph_filter.py для деталей.
            result = _drop_morph_case_substitutions(result, raw_text)
            # Финальная валидация: дропаем пункты, чьё «было» отсутствует
            # в raw_text (галлюцинации модели — например, yandex-corrector
            # пишет «безопасностей» там, где в тексте «безопасности»).
            # Без этого фильтра клиент покажет «фрагмент не найден».
            result = _drop_changes_not_in_text(result, raw_text)
            # Если модель отдала правильный CORRECTED, но все её пункты
            # CHANGES оказались галлюцинациями (выкинуты выше) — реконструируем
            # пункты CHANGES из diff(raw_text, CORRECTED). Это ключевой фикс
            # для yandex-corrector: она хорошо корректирует текст, но плохо
            # рапортует, и без этой ветки пользователь видит «Ошибок не
            # найдено», хотя в CORRECTED уже стоят все нужные запятые.
            if had_pairs_pre_filter and not _has_real_change_items(result):
                corrected_body = _extract_corrected_body(result)
                if corrected_body and corrected_body != raw_text:
                    rebuilt = _rebuild_changes_from_diff(raw_text, corrected_body)
                    if rebuilt:
                        logger.info(
                            "Реконструировано %d пункт(ов) CHANGES из diff "
                            "(модель забыла рапорт)", len(rebuilt),
                        )
                        result = _replace_changes_block(result, rebuilt)
            # v1.8b: дропаем пункты, в которых модель «исправляет»
            # whitelisted-термины пользовательского словаря (ЦСН, КС-2,
            # имена организаций и т.п.). Если словарь пуст — no-op.
            result = _drop_user_dict_changes(result)
            # v1.8a: обогащаем CHANGES пунктами, которые морфо-детектор
            # нашёл в raw_text (numeral-noun «во 2-м кварталах»,
            # adj-noun «капитальных ремонтова», и т.п.). Если детектор
            # отключён или ошибок не найдено — no-op. Дедупликация по
            # `before` чтобы не дублировать пункты, уже отданные моделью.
            result = _enrich_changes_with_detector(result, raw_text)
            # v1.8c: пост-валидация sage-95m. Если SAGE_VALIDATOR_ENABLED=false
            # или модель не загрузилась — no-op. Идёт ПОСЛЕ детектора, чтобы
            # sage оценивал ТОЛЬКО правки от LLM (детектор уже отфильтрован
            # по precision на уровне правил). Latency cost ~1-3 с warm.
            result = _filter_changes_with_sage(result, raw_text)
            # v2.0-b: параллельный детектор LanguageTool-RU. Добавляет
            # стилистические + типографские правки, которые T-lite и
            # MorphDetector обычно не трогают (тире вместо дефиса,
            # русские кавычки, повторы слов, канцеляризмы и т.п.).
            # Если LT-сервер недоступен или LANGUAGETOOL_ENABLED=false —
            # no-op. Идёт ПОСЛЕ sage, т.к. sage валидирует только LLM-
            # правки, а LT-правки — rule-based, ML-проверка не нужна.
            result = _enrich_changes_with_languagetool(result, raw_text)
            # v1.8.4: закрываем рассинхрон CHANGES↔CORRECTED. T-lite иногда
            # склеивает несколько правок (орфография+согласование) в один
            # пункт CHANGES, а CORRECTED показывает обе. Клиент применит
            # только пункт CHANGES — получит частично-исправленный текст.
            # Эта функция симулирует применение CHANGES к raw_text, сверяет
            # с CORRECTED и при desync дописывает недостающие правки в
            # конец CHANGES (через diff). CORRECTED не трогает.
            result = _complete_changes_from_corrected(result, raw_text)
            # v1.7.3: финальная пере-нумерация CHANGES (после всех drop'ов
            # и возможной реконструкции). Закрывает кейс «правки начались
            # со 2го пункта» в LibreOffice-расширении: если фильтр дропнул
            # пункт 1 — нужно перенумеровать оставшиеся, чтобы клиент
            # видел сплошную нумерацию 1, 2, 3, а не 2, 3.
            result = _renumber_changes(result)
        except Exception as e:
            ok = False
            error = f"{type(e).__name__}: {e}"
            logger.exception("Ошибка запроса к Ollama")
            result = f"ОШИБКА_СЕРВЕРА: {error}"

    audit.record(
        client_ip=client_ip, user_agent=user_agent,
        server="local", model=MODEL_NAME,
        text=raw_text, context=raw_ctx,
        changes_count=count_changes(result),
        duration_ms=timer.ms, ok=ok, error=error,
    )
    logger.info(
        "suggest ip=%s len=%d ctx=%d changes=%d ok=%s dur=%dms",
        client_ip, len(raw_text), len(raw_ctx),
        count_changes(result), ok, timer.ms,
    )
    return result
