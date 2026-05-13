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


_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Срезает <think>…</think> и leading-рассуждения, если модель проигнорировала /no_think.

    Возвращает «чистый» ответ. Если в тексте нет ни тегов <think>, ни маркера
    ===CORRECTED===, не трогаем — пусть верхний слой сам разбирается.
    """
    cleaned = _THINK_BLOCK.sub("", text)
    # Иногда qwen3 без тегов пишет рассуждения, а ===CORRECTED=== всё равно есть ниже.
    idx = cleaned.find("===CORRECTED===")
    if idx > 0:
        cleaned = cleaned[idx:]
    return cleaned.strip()


# Угловые/типографские кавычки, встречающиеся в ===CHANGES===. Одиночные
# ' и ` не включаем — они ложно срабатывают на апостроф/транслитерацию.
_QUOTE_CHARS = "«»\"“”‘’‚‛„"
# Разделитель между «было» и «стало». Допускаем стрелки (→, ->), тире
# (—, –, -) и текстовые связки, включая обороты «заменено/исправлено на».
# В сепараторе разрешаем любые символы, кроме кавычек — так захватываются
# варианты вроде «X» — исправлено на «X» или «X» заменено на «X».
_CHANGE_PAIR_RE = re.compile(
    rf"[{_QUOTE_CHARS}]([^{_QUOTE_CHARS}]+)[{_QUOTE_CHARS}]"
    rf"[^{_QUOTE_CHARS}]*?"
    rf"[{_QUOTE_CHARS}]([^{_QUOTE_CHARS}]+)[{_QUOTE_CHARS}]",
    re.IGNORECASE,
)


_LEADING_NUM_RE = re.compile(r"^\s*\d+\.\s*")
_QUOTED_GREEDY_RE = re.compile(
    rf"^\s*[{_QUOTE_CHARS}](.+)[{_QUOTE_CHARS}]\s*$"
)


def _parse_change_pair_robust(line: str) -> Optional[tuple[str, str]]:
    """Извлекает (before, after) из строки CHANGES, корректно обрабатывая
    вложенные кавычки. v1.8.2: _CHANGE_PAIR_RE использует non-greedy
    срез между [^кавычек]+, и на строке вида «адм…здания «ЦСН ВО»» →
    «адм…здания «ЦСН ВО»» он матчится на внутренней «ЦСН ВО», а не на
    внешней парe — в итоге before/after не равны и фильтр пропускает
    идемпотентный пункт.

    Эта функция работает по структуре строки:
      `N. «before» → «after» | explanation` → (before, after).
    Жадный матч `«(.+)»` забирает всё содержимое от первой « до последней ».
    """
    s = _LEADING_NUM_RE.sub("", line.strip())
    # Отрезаем ` | explanation` (если есть)
    if " | " in s:
        s = s.split(" | ", 1)[0]
    # Ищем разделитель before/after
    for sep in (" → ", " -> "):
        if sep in s:
            left, right = s.rsplit(sep, 1)
            lm = _QUOTED_GREEDY_RE.match(left.strip())
            rm = _QUOTED_GREEDY_RE.match(right.strip())
            if lm and rm:
                return (lm.group(1), rm.group(1))
            return None
    return None


def _drop_idempotent_changes(text: str) -> str:
    """Удаляет из блока ===CHANGES=== пункты вида «X → X».

    Некоторые модели (в частности T-lite-it-2.1) на задаче корректуры иногда
    перечисляют в changelog правила из системного промпта, выдавая пустые
    пункты типа «согласно распоряжению → согласно распоряжению» или
    «отдел подготовил отчётность → отдел подготовил отчётность» там, где
    исправлений не было. Такие пункты бесполезны для пользователя и
    засоряют Track Changes. Удаляем их.

    Если после фильтрации в ===CHANGES=== не осталось ни одного пункта —
    подставляем заглушку «Ошибок не найдено».
    """
    if "===CHANGES===" not in text or "===END===" not in text:
        return text
    try:
        before, rest = text.split("===CHANGES===", 1)
        changes_block, tail = rest.split("===END===", 1)
    except ValueError:
        return text

    kept: list[str] = []
    for raw_line in changes_block.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            kept.append(line)
            continue
        # v1.8.2: сначала пробуем robust-парсер (учитывает вложенные кавычки),
        # потом старый _CHANGE_PAIR_RE как fallback.
        pair = _parse_change_pair_robust(line)
        if pair is None:
            m = _CHANGE_PAIR_RE.search(line)
            if m:
                pair = (m.group(1), m.group(2))
        # Сравниваем БЕЗ .lower(): «Приказа» → «приказа» — это валидная
        # орфографическая правка регистра (имя собственное vs нарицательное),
        # такие пункты сохраняем. Идемпотентный пункт — это когда до и после
        # совпадают побуквенно.
        if pair and pair[0].strip() == pair[1].strip():
            logger.debug("Фильтрую идемпотентный пункт: %s", line.strip())
            continue
        # Пункты с многоточием в цитатах («…» или «...») неприменимы:
        # клиент ищет «было» через InStr в выделении, а сокращённую цитату
        # никогда не найдёт. Лучше скрыть пункт, чем показывать пользователю
        # «не удалось применить (фрагмент не найден)». YandexGPT-5-Lite
        # склонна к таким сокращениям; T-lite — реже. Промпт это запрещает,
        # но оставляем как страховку.
        if pair and ("…" in pair[0] or "..." in pair[0]
                     or "…" in pair[1] or "..." in pair[1]):
            logger.info("Фильтрую пункт с многоточием в цитате: %s", line.strip())
            continue
        kept.append(line)

    # Есть ли хотя бы один пронумерованный пункт с текстом?
    non_empty = [ln for ln in kept if re.search(r"\w", ln)]
    has_real_item = any(re.match(r"\s*\d+\.\s*\S", ln) for ln in non_empty)
    if not has_real_item:
        kept = ["", "1. Ошибок не найдено. Текст соответствует нормам.", ""]

    new_changes = "\n".join(kept).rstrip() + "\n"
    return f"{before}===CHANGES===\n{new_changes.lstrip()}===END==={tail}"


_CHANGE_NUM_RE = re.compile(r"^(\s*)(\d+)\.(\s*)(.*)$")


def _filter_changes_with_sage(text: str, raw_text: str) -> str:
    """v1.8c: пост-валидация ===CHANGES=== через sage-fredt5-distilled-95m.

    Алгоритм:
      1. Sage один раз корректирует raw_text → sage_corrected.
      2. Для каждой пары (before → after) из CHANGES берём verdict у
         SageValidator.judge():
           - AGREE: sage воспроизвёл `after` → правку оставляем.
           - DISAGREE: sage оставил `before` → возможно FP T-lite.
           - UNKNOWN: sage сделал что-то третье → неоднозначно.
      3. По domain'у решаем, дропать или нет:
           - admin: дропаем только DISAGREE (recall важнее).
           - general: дропаем DISAGREE и UNKNOWN.
      4. Если правка дропнута — пытаемся откатить её и в CORRECTED-блоке
         (заменой первого вхождения `after` обратно на `before`).
         Это критично для консистентности: иначе пользователь видит
         CORRECTED с правкой, которую нельзя применить — пункт CHANGES
         удалён.

    No-op если sage недоступен, отключён или CHANGES пуст.
    """
    if _sage_validator is None or not _sage_validator.is_available():
        return text
    if not raw_text or not raw_text.strip():
        return text
    if "===CHANGES===" not in text or "===END===" not in text:
        return text
    if "===CORRECTED===" not in text:
        return text

    # Один прогон sage на исходный текст. Если упадёт — return text.
    try:
        sage_text = _sage_validator.correct(raw_text)
    except Exception as e:  # noqa: BLE001
        logger.warning("Sage: ошибка correct(), пропускаю валидацию: %s", e)
        return text
    if not sage_text or sage_text == raw_text:
        # Sage не нашёл ошибок — это сильный сигнал, что T-lite, возможно,
        # перестарался. НО для admin-режима не доверяем «sage молчит ==
        # ошибок нет» — sage-95m модель компактная, может пропускать.
        # Поэтому в admin-режиме при пустом изменении sage'a не делаем
        # ничего (не дропаем все правки). В general — тоже не дропаем
        # автоматически, чтобы избежать catastrophic-фильтра. Sage
        # используется только для **позитивной проверки** конкретной
        # правки, не для отрицательного приговора всему документу.
        return text

    # Извлекаем блоки
    try:
        before_block, rest = text.split("===CHANGES===", 1)
        changes_block, tail = rest.split("===END===", 1)
        # CORRECTED — между «===CORRECTED===» и «===CHANGES===»
        corrected_start = before_block.find("===CORRECTED===")
        if corrected_start < 0:
            return text
        head = before_block[:corrected_start]
        corrected_full = before_block[corrected_start:]
        corrected_body_match = re.match(
            r"===CORRECTED===\s*\n?(.*)\Z", corrected_full, flags=re.DOTALL
        )
        if corrected_body_match is None:
            return text
        corrected_body = corrected_body_match.group(1)
    except ValueError:
        return text

    kept_lines: list[str] = []
    dropped = 0
    new_corrected = corrected_body

    for raw_line in changes_block.splitlines():
        line = raw_line.rstrip()
        pair = _parse_change_pair_robust(line)
        if pair is None:
            m = _CHANGE_PAIR_RE.search(line)
            if m:
                pair = (m.group(1), m.group(2))
        if pair is None:
            # Не пункт-CHANGE (заголовок/комментарий/пустая) — сохраняем
            kept_lines.append(line)
            continue

        before_q, after_q = pair[0], pair[1]
        # Извлекаем «категорию» — это всё после первого «|» в строке
        # (формат: «X → Y | орфография — пропущена буква…»). Используется
        # для category-aware фильтрации: sage обучена на орфографии,
        # для согласования/управления — ненадёжна.
        category = ""
        if "|" in line:
            category = line.split("|", 1)[1].strip()
        verdict = _sage_validator.judge(before_q, after_q, sage_text)
        # Логируем verdict ВСЕГДА (даже в dryrun) — это основной способ
        # собрать прод-данные перед переключением в enforce.
        logger.info(
            "Sage[%s/%s/cat=%r]: verdict=%s для %r→%r",
            _sage_validator.config.mode, _sage_validator.config.domain,
            category[:40], verdict, before_q, after_q,
        )
        if _sage_validator.should_drop(verdict, category=category):
            logger.info(
                "Sage[enforce]: ДРОП правки %r→%r (verdict=%s, cat=%r)",
                before_q, after_q, verdict, category[:40],
            )
            dropped += 1
            # Пробуем откатить правку в CORRECTED.  Заменяем первое
            # вхождение `after_q` обратно на `before_q`. Если `after_q`
            # не находится (whitespace shift / multiword разница) —
            # пропускаем revert, оставляем CORRECTED как есть. В таком
            # случае пункт всё равно дропнут из CHANGES, но CORRECTED
            # останется частично «исправленным» — для admin-режима это
            # приемлемо: пользователь увидит CORRECTED с правкой, и
            # сам решит, применять её или нет.
            idx = new_corrected.find(after_q)
            if idx >= 0:
                new_corrected = (
                    new_corrected[:idx] + before_q + new_corrected[idx + len(after_q):]
                )
            continue
        kept_lines.append(line)

    if dropped == 0:
        return text

    # Если все правки дропнуты — стандартная заглушка
    non_empty = [ln for ln in kept_lines if re.search(r"\w", ln)]
    has_real = any(re.match(r"\s*\d+\.\s*\S", ln) for ln in non_empty)
    if not has_real:
        kept_lines = ["", "1. Ошибок не найдено. Текст соответствует нормам.", ""]

    new_changes = "\n".join(kept_lines).rstrip() + "\n"
    new_corrected_block = "===CORRECTED===\n" + new_corrected.lstrip("\n")
    if not new_corrected_block.endswith("\n"):
        new_corrected_block += "\n"
    return (
        f"{head}{new_corrected_block}"
        f"===CHANGES===\n{new_changes.lstrip()}===END==={tail}"
    )


def _complete_changes_from_corrected(text: str, raw_text: str) -> str:
    """v1.8.4: закрывает рассинхрон ===CHANGES=== ↔ ===CORRECTED===.

    Симулирует применение CHANGES к raw_text (как это делает клиент
    LibreOffice через InStr+Replace) и сверяет результат с CORRECTED.
    Если CORRECTED содержит правки, не покрытые ни одним пунктом
    CHANGES — добавляет недостающие правки в конец CHANGES через diff.

    Реальный прод-кейс (v1.8c прогон, 05.05.2026): T-lite склеила две
    правки «капитальных → капитального» (согласование) и «ремонтова
    → ремонта» (орфография) в один пункт `«ремонтова» → «ремонта»`.
    CORRECTED содержит «капитального ремонта», но CHANGES — только
    орфо-правку. Клиент применит CHANGES → получит «капитальных
    ремонта» (несогласованно). Эта функция добавляет пункт
    «капитальных → капитального» через diff(simulated, corrected).

    Не модифицирует CORRECTED — только дополняет CHANGES до полного
    покрытия. Если diff не находит новых правок — no-op.

    Идёт ПОСЛЕ всех фильтров (drop_idempotent, drop_eyo, morph_filter,
    drop_changes_not_in_text, drop_user_dict, enrich_with_detector,
    filter_with_sage) и ДО _renumber_changes (новые пункты получат
    последовательные номера).
    """
    if not text or not raw_text:
        return text
    if "===CHANGES===" not in text or "===END===" not in text:
        return text
    if "===CORRECTED===" not in text:
        return text

    corrected_body = _extract_corrected_body(text)
    if not corrected_body or corrected_body == raw_text:
        return text

    try:
        head, rest = text.split("===CHANGES===", 1)
        changes_block, tail = rest.split("===END===", 1)
    except ValueError:
        return text

    # Парсим существующие (before, after) пары из CHANGES.  Сохраняем
    # порядок — нужно для корректной симуляции применения.
    existing_pairs: list[tuple[str, str]] = []
    for raw_line in changes_block.splitlines():
        line = raw_line.strip()
        if not line or "Ошибок не найдено" in line:
            continue
        pair = _parse_change_pair_robust(line)
        if pair is None:
            m = _CHANGE_PAIR_RE.search(line)
            if m:
                pair = (m.group(1), m.group(2))
        if pair is None:
            continue
        before_q, after_q = pair[0].strip(), pair[1].strip()
        if not before_q or not after_q or before_q == after_q:
            continue
        existing_pairs.append((before_q, after_q))

    # Симулируем то, что сделает клиент: для каждого пункта ищем первое
    # вхождение `before` в текущем тексте и заменяем на `after`. Если
    # `before` не найден — пропускаем (клиент покажет «фрагмент не
    # найден», но это уже отфильтровано _drop_changes_not_in_text).
    simulated = raw_text
    for before_q, after_q in existing_pairs:
        idx = simulated.find(before_q)
        if idx < 0:
            continue
        simulated = simulated[:idx] + after_q + simulated[idx + len(before_q):]

    # Нормализуем хвосты пробелов/переносов для сравнения — клиент тоже
    # сохраняет finalNewline, а CORRECTED-блок может содержать лишние
    # переносы по краям.
    if simulated.strip() == corrected_body.strip():
        return text

    # diff(simulated → corrected_body) даст ровно те правки, которых
    # не хватает в CHANGES для полного воспроизведения CORRECTED.
    missing_entries = _rebuild_changes_from_diff(simulated, corrected_body)
    if not missing_entries:
        return text

    # Дедупликация: исключаем пункты, чей `before` уже есть в CHANGES
    # (защита от двойного эмита, если diff поймал перекрытие).
    existing_befores = {b.strip().lower() for b, _ in existing_pairs}
    new_entries: list[str] = []
    for entry in missing_entries:
        m = _CHANGE_PAIR_RE.search(entry)
        if m is None:
            continue
        b_norm = m.group(1).strip().lower()
        if b_norm in existing_befores:
            continue
        new_entries.append(entry)
        existing_befores.add(b_norm)

    if not new_entries:
        return text

    logger.info(
        "v1.8.4: добавлено %d пункт(ов) CHANGES для синхронизации с "
        "CORRECTED (T-lite не перечислил все правки)", len(new_entries),
    )

    # Приклеиваем новые пункты в конец changes_block (с переносами).
    # _renumber_changes ниже нормализует нумерацию подряд (1, 2, 3...).
    suffix = "\n".join(new_entries)
    existing_kept = changes_block.rstrip("\n")
    # Если в блоке остался только стаб «Ошибок не найдено» — затираем
    # его, теперь у нас реальные пункты.
    if existing_kept.strip() and "Ошибок не найдено" in existing_kept:
        existing_kept = ""
    if existing_kept:
        new_changes = existing_kept + "\n" + suffix + "\n"
    else:
        new_changes = "\n" + suffix + "\n"
    return f"{head}===CHANGES==={new_changes}===END==={tail}"


def _renumber_changes(text: str) -> str:
    """v1.7.3: пере-нумеровывает пункты ===CHANGES=== подряд (1, 2, 3...)
    после того как ранее работающие фильтры (`_drop_idempotent_changes`,
    `_drop_changes_not_in_text`, `_drop_eyo_substitutions`,
    `_drop_morph_case_substitutions`) могли удалить часть пунктов и
    оставить «дырки» в нумерации (например, «2. ... 4. ...»).

    Пользователь в LibreOffice-расширении видел: «правки начались со
    2го пункта». Это происходит когда модель отдала «1. X 2. Y 3. Z»,
    фильтр дропнул «1. X», осталось «2. Y 3. Z». Эта функция превратит
    их обратно в «1. Y 2. Z».

    Не трогает строки без пронумерованного префикса (пустые строки,
    «Ошибок не найдено» как стаб, etc.). Не меняет порядок пунктов.
    """
    if "===CHANGES===" not in text or "===END===" not in text:
        return text
    try:
        before, rest = text.split("===CHANGES===", 1)
        changes_block, tail = rest.split("===END===", 1)
    except ValueError:
        return text
    new_lines: list[str] = []
    next_num = 1
    for raw_line in changes_block.splitlines():
        line = raw_line.rstrip()
        m = _CHANGE_NUM_RE.match(line)
        if m:
            indent, _old_num, sep, content = m.group(1), m.group(2), m.group(3), m.group(4)
            new_lines.append(f"{indent}{next_num}.{sep}{content}")
            next_num += 1
        else:
            new_lines.append(line)
    new_changes = "\n".join(new_lines).rstrip() + "\n"
    return f"{before}===CHANGES===\n{new_changes.lstrip()}===END==={tail}"


def _drop_user_dict_changes(text: str) -> str:
    """v1.8b: Дропает пункты ===CHANGES===, в которых модель пытается
    «исправить» whitelisted-термин из пользовательского словаря.

    Логика: если в `before` или `after` пункта встречается слово из
    словаря (case-insensitive, по границе слова), пункт удаляется.
    Это защищает от регрессии случаев типа «ЦСН → ЦНС», «КС-2 → КС2»
    которые юзер уже однажды пометил как корректные.

    Если `_user_dict` не инициализирован или пуст — no-op.
    """
    if _user_dict is None:
        return text
    words = _user_dict.list_words()
    if not words:
        return text
    if "===CHANGES===" not in text or "===END===" not in text:
        return text
    try:
        before_block, rest = text.split("===CHANGES===", 1)
        changes_block, tail = rest.split("===END===", 1)
    except ValueError:
        return text
    # Для производительности: один регэксп всех whitelisted-слов через alternation
    word_patterns = [re.escape(w) for w in words]
    word_re = re.compile(
        r"(?<![\w-])(?:" + "|".join(word_patterns) + r")(?![\w-])",
        re.IGNORECASE,
    )
    kept: list[str] = []
    dropped = 0
    for line in changes_block.splitlines():
        m = _CHANGE_PAIR_RE.search(line)
        if m:
            before_q, after_q = m.group(1), m.group(2)
            if word_re.search(before_q) or word_re.search(after_q):
                logger.info(
                    "Дроп правки whitelisted-термина (user_dict): «%s» → «%s»",
                    before_q.strip(), after_q.strip(),
                )
                dropped += 1
                continue
        kept.append(line)
    if dropped == 0:
        return text
    # Если ничего не осталось — стандартное «Ошибок не найдено»
    non_empty = [ln for ln in kept if re.search(r"\w", ln)]
    has_real = any(re.match(r"\s*\d+\.\s*\S", ln) for ln in non_empty)
    if not has_real:
        kept = ["", "1. Ошибок не найдено. Текст соответствует нормам.", ""]
    new_changes = "\n".join(kept).rstrip() + "\n"
    return f"{before_block}===CHANGES===\n{new_changes.lstrip()}===END==={tail}"


def _enrich_changes_with_detector(text: str, raw_text: str) -> str:
    """v1.8a: Запускает morph_detector на raw_text и добавляет в CHANGES
    пункты, которых модель не дала.

    Если детектор отключён или не нашёл ошибок — no-op. Дедупликация:
    пункты с тем же `before` (case-insensitive) что уже есть в CHANGES,
    не добавляются повторно.

    Также применяется CORRECTED-патч: если детектор нашёл numeral_noun
    или adj_noun ошибку, его suggestion применяется к CORRECTED-блоку
    (заменой первого вхождения `before` на `suggestion`). Это позволяет
    клиенту LibreOffice применить правку через diff. OOV-ошибки с
    пустым suggestion не патчат CORRECTED — их пользователь правит сам.
    """
    if _morph_detector is None or not _morph_detector.available:
        return text
    if not raw_text:
        return text
    if "===CHANGES===" not in text or "===END===" not in text:
        return text
    if "===CORRECTED===" not in text:
        return text
    # Детектор: получаем список GrammarError, фильтруем whitelist'ом
    whitelist = _user_dict.as_frozenset() if _user_dict is not None else None
    try:
        errors = _morph_detector.detect_errors(raw_text, whitelist=whitelist)
    except Exception as exc:
        logger.warning("MorphDetector упал на raw_text (size=%d): %s", len(raw_text), exc)
        return text
    if not errors:
        return text
    # Извлекаем существующие `before` из CHANGES для дедупликации
    try:
        before_block, rest = text.split("===CHANGES===", 1)
        changes_block, tail = rest.split("===END===", 1)
    except ValueError:
        return text
    existing_befores: set[str] = set()
    for line in changes_block.splitlines():
        m = _CHANGE_PAIR_RE.search(line)
        if m:
            existing_befores.add(m.group(1).strip().lower())
    # Извлекаем CORRECTED тело
    corrected_body = _extract_corrected_body(text) or raw_text
    new_lines: list[str] = []
    added_pairs = 0
    patched_corrected = corrected_body
    for err in errors:
        if err.before.strip().lower() in existing_befores:
            continue
        # Не дублируем тот же before если уже добавлен детектором
        if err.before.strip().lower() in {e.split("«")[1].split("»")[0].lower() for e in new_lines if "«" in e}:
            continue
        # Если у ошибки есть suggestion и он отличается от before —
        # формируем нормальную «X → Y» строку. Иначе (OOV без suggestion)
        # пишем «X» → «X» с пометкой "проверьте написание" — НЕТ, это
        # будет дропнуто `_drop_idempotent_changes`. Поэтому для OOV
        # просто не добавляем CHANGES, а только лог пишем.
        if not err.suggestion or err.suggestion == err.before:
            logger.info(
                "MorphDetector: OOV-слово в raw_text «%s» (offset=%d) — не добавляем в CHANGES (suggestion отсутствует)",
                err.before, err.offset,
            )
            continue
        new_lines.append(err.to_change_line(0))  # number перенумеруется ниже
        existing_befores.add(err.before.strip().lower())
        added_pairs += 1
        # Патч CORRECTED — заменяем first occurrence
        if err.before in patched_corrected:
            patched_corrected = patched_corrected.replace(err.before, err.suggestion, 1)
    if added_pairs == 0:
        return text
    # Соединяем существующие + новые пункты
    existing_kept = changes_block.rstrip()
    if existing_kept and "Ошибок не найдено" in existing_kept:
        # Если был стаб — затираем его, т.к. теперь у нас реальные пункты
        existing_kept = ""
    if existing_kept:
        merged_changes = existing_kept + "\n" + "\n".join(new_lines) + "\n"
    else:
        merged_changes = "\n" + "\n".join(new_lines) + "\n"
    logger.info(
        "MorphDetector: добавлено %d пункт(ов) в CHANGES "
        "(пропущено моделью)", added_pairs,
    )
    # Заменяем CORRECTED тело и CHANGES блок
    new_text = before_block + "===CHANGES===" + merged_changes + "===END===" + tail
    if patched_corrected != corrected_body:
        new_text = _replace_corrected_body(new_text, patched_corrected)
    return new_text


def _replace_corrected_body(text: str, new_body: str) -> str:
    """Заменяет содержимое блока ===CORRECTED===…===CHANGES=== на new_body."""
    if "===CORRECTED===" not in text or "===CHANGES===" not in text:
        return text
    try:
        before, rest = text.split("===CORRECTED===", 1)
        _, tail = rest.split("===CHANGES===", 1)
    except ValueError:
        return text
    return f"{before}===CORRECTED===\n{new_body.strip()}\n===CHANGES==={tail}"


def _drop_changes_not_in_text(text: str, raw_text: str) -> str:
    """Дропает пункты ===CHANGES===, чьё «было» не является подстрокой
    исходного текста пользователя.

    Это страховка от галлюцинаций модели: если модель пишет
    `«безопасностей» → «безопасности»`, но в `raw_text` слова
    «безопасностей» нет — пункт неприменим, клиент покажет «фрагмент
    не найден». Лучше сразу скрыть на сервере.

    Сравнение строгое (substring), без нормализации. Это сознательно:
    клиент тоже ищет через InStr строгим сравнением, и если на сервере
    пункт прошёл, то и на клиенте найдётся.

    Если после фильтрации не осталось ни одного пункта — подставляем
    «Ошибок не найдено» (как в _drop_idempotent_changes)."""
    if "===CHANGES===" not in text or "===END===" not in text:
        return text
    if not raw_text:
        return text
    try:
        before, rest = text.split("===CHANGES===", 1)
        changes_block, tail = rest.split("===END===", 1)
    except ValueError:
        return text

    kept: list[str] = []
    dropped_count = 0
    for raw_line in changes_block.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            kept.append(line)
            continue
        m = _CHANGE_PAIR_RE.search(line)
        if m:
            quote_before = m.group(1).strip()
            # Пункт неприменим, если «было» отсутствует в исходном тексте.
            # Пустые цитаты («» → «X», вставка) считаем невалидными тоже.
            if not quote_before or quote_before not in raw_text:
                logger.info(
                    "Дроп пункта: «%s» нет в raw_text (галлюцинация): %s",
                    quote_before, line.strip(),
                )
                dropped_count += 1
                continue
        kept.append(line)

    non_empty = [ln for ln in kept if re.search(r"\w", ln)]
    has_real_item = any(re.match(r"\s*\d+\.\s*\S", ln) for ln in non_empty)
    if not has_real_item:
        kept = ["", "1. Ошибок не найдено. Текст соответствует нормам.", ""]

    if dropped_count:
        logger.info("Отфильтровано %d пункт(ов) с галлюцинированным «было»",
                    dropped_count)

    new_changes = "\n".join(kept).rstrip() + "\n"
    return f"{before}===CHANGES===\n{new_changes.lstrip()}===END==={tail}"


def _is_eyo_only_substitution(before: str, after: str) -> bool:
    """True, если `before` и `after` отличаются ИСКЛЮЧИТЕЛЬНО подменой ё↔е
    (или Ё↔Е). Это стилистика, а не орфографическая ошибка: «проведенными»
    и «проведёнными» — обе формы нормативны (ё в русском факультативно
    обозначается на письме). Модель не должна такие правки выдавать."""
    if not before or not after or before == after:
        return False
    norm_before = before.replace("ё", "е").replace("Ё", "Е")
    norm_after = after.replace("ё", "е").replace("Ё", "Е")
    return norm_before == norm_after


def _drop_eyo_substitutions(text: str, raw_text: str) -> str:
    """Дропает пункты ===CHANGES===, отличающиеся только заменой ё↔е,
    и откатывает подмену в ===CORRECTED===, восстанавливая оригинальное
    написание буквы из `raw_text`.

    Это страховка от стабильной галлюцинации T-lite-it-2.1: модель любит
    «улучшать» е → ё в причастиях («проведенными» → «проведёнными»,
    «повлекших» → «повлёкших»), хотя по правилам РАН употребление ё
    факультативно. v1.6.7 пробовал запретить это в SYSTEM_PROMPT — не
    помогло. v1.6.8: фильтруем после генерации.

    Срабатывает только если `before` есть в `raw_text` (т.е. правка
    реально была применена). Чтобы не ломать редкие случаи, когда ё
    действительно нужна (имена типа «Тёркин» — но они не должны быть в
    «исправлениях» вообще).
    """
    if "===CORRECTED===" not in text or "===CHANGES===" not in text:
        return text
    if "===END===" not in text:
        return text
    if not raw_text:
        return text
    try:
        head, rest1 = text.split("===CORRECTED===", 1)
        corrected_block, rest2 = rest1.split("===CHANGES===", 1)
        changes_block, tail = rest2.split("===END===", 1)
    except ValueError:
        return text

    new_corrected = corrected_block
    kept: list[str] = []
    dropped_count = 0
    for raw_line in changes_block.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            kept.append(line)
            continue
        m = _CHANGE_PAIR_RE.search(line)
        if m:
            quote_before = m.group(1).strip()
            quote_after = m.group(2).strip()
            if (
                _is_eyo_only_substitution(quote_before, quote_after)
                and quote_before in raw_text
                and quote_after in new_corrected
            ):
                # Откатываем подмену в CORRECTED — заменяем ВСЕ вхождения,
                # потому что модель могла «исправить» одно и то же слово
                # несколько раз в разных предложениях.
                new_corrected = new_corrected.replace(quote_after, quote_before)
                logger.info(
                    "Дроп стилистической ё-замены: «%s» → «%s» "
                    "(откат в CORRECTED)",
                    quote_before, quote_after,
                )
                dropped_count += 1
                continue
        kept.append(line)

    if dropped_count == 0:
        return text

    non_empty = [ln for ln in kept if re.search(r"\w", ln)]
    has_real_item = any(re.match(r"\s*\d+\.\s*\S", ln) for ln in non_empty)
    if not has_real_item:
        kept = ["", "1. Ошибок не найдено. Текст соответствует нормам.", ""]

    logger.info("Отфильтровано %d пункт(ов) ё-замены (стилистика, не ошибка)",
                dropped_count)

    new_changes = "\n".join(kept).rstrip() + "\n"
    return (
        f"{head}===CORRECTED==={new_corrected}"
        f"===CHANGES===\n{new_changes.lstrip()}===END==={tail}"
    )


def _undo_eyo_in_text(corrected: str, raw_text: str) -> str:
    """Посимвольно откатывает в `corrected` подмены ё→е (и Ё→Е), которые
    модель сделала «улучшая» написание, но которых нет в `raw_text`.

    Закрывает bypass `_drop_eyo_substitutions`: модель часто упаковывает
    ё-подмену вместе с реальной правкой в одну compound-цитату вида
    «повлекших риски ... Подразделения» → «повлёкших риски ... Подразделению»;
    line-level фильтр такой пункт оставляет (внутри есть и НЕ-ё разница),
    но в CORRECTED всё равно стоит «повлёкших». Этот фильтр выровнивает
    raw_text и corrected по символам через difflib и для каждой замены
    («replace» opcode) с одинаковой длиной сегмента — посимвольно
    восстанавливает букву из raw_text там, где разница только ё↔е.

    Сегменты разной длины не трогаем — это места реальных правок (вставки
    или удаления символов).
    """
    if "ё" not in corrected and "Ё" not in corrected:
        return corrected
    if not raw_text:
        return corrected
    matcher = difflib.SequenceMatcher(None, raw_text, corrected, autojunk=False)
    parts: list[str] = []
    undone = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(corrected[j1:j2])
        elif tag == "replace":
            raw_seg = raw_text[i1:i2]
            corr_seg = corrected[j1:j2]
            if len(raw_seg) == len(corr_seg):
                fixed_chars: list[str] = []
                for r_ch, c_ch in zip(raw_seg, corr_seg):
                    if c_ch == "ё" and r_ch == "е":
                        fixed_chars.append("е")
                        undone += 1
                    elif c_ch == "Ё" and r_ch == "Е":
                        fixed_chars.append("Е")
                        undone += 1
                    else:
                        fixed_chars.append(c_ch)
                parts.append("".join(fixed_chars))
            else:
                parts.append(corr_seg)
        elif tag == "insert":
            parts.append(corrected[j1:j2])
        # tag == "delete": в `corrected` ничего нет — пропускаем.
    if undone == 0:
        return corrected
    logger.info(
        "Откат ё→е в CORRECTED: %d символ(ов) восстановлено по raw_text",
        undone,
    )
    return "".join(parts)


def _undo_eyo_in_corrected_block(text: str, raw_text: str) -> str:
    """Применяет `_undo_eyo_in_text` к содержимому ===CORRECTED===
    в полном ответе модели. Не трогает ===CHANGES===."""
    if "===CORRECTED===" not in text or "===CHANGES===" not in text:
        return text
    try:
        head, rest = text.split("===CORRECTED===", 1)
        corrected_block, tail = rest.split("===CHANGES===", 1)
    except ValueError:
        return text
    new_corrected = _undo_eyo_in_text(corrected_block, raw_text)
    if new_corrected == corrected_block:
        return text
    return f"{head}===CORRECTED==={new_corrected}===CHANGES==={tail}"


def _drop_morph_case_substitutions(text: str, raw_text: str) -> str:
    """v1.7: дропает пункты ===CHANGES===, представляющие собой
    «улучшение» падежной формы валидного слова без реального
    грамматического основания, и откатывает подмену в ===CORRECTED===.

    Работает только если `_morph_filter` инициализирован и `available`
    (pymorphy3 загружен). Иначе — no-op, чтобы PR можно было откатить
    через простое удаление зависимости.

    Главный prod-кейс (КС-2, 5 мая 2026): «Подразделения» (родительный
    падеж) → «Подразделению» (дательный) — обе формы валидны, контекст
    «причинения ущерба Подразделения» НЕ требует именно дательного.
    Модель «улучшает» по common pattern «ущерб + dat», но в нашем
    тексте «Подразделения» — это родительный принадлежности, и правка
    меняет смысл. Char-level eyo undo здесь бессилен (ё нет), и
    `_drop_changes_not_in_text` тоже (Подразделения ЕСТЬ в raw_text).

    Логика «когда дроп»:
      * `_morph_filter.is_hallucinated_case_change(before, after, raw_text)` →
        True (одна лемма, тот же number, разный case, перед `before` нет
        управляющего предлога).

    Чего НЕ трогает (защита от false positive):
      * лексические замены (разные леммы);
      * число различается — это agreement fix, реальная правка;
      * перед `before` стоит case-governing предлог («согласно приказа» →
        «согласно приказу» — реальная ошибка управления).
    """
    if _morph_filter is None or not _morph_filter.available:
        return text
    if "===CORRECTED===" not in text or "===CHANGES===" not in text:
        return text
    if "===END===" not in text:
        return text
    if not raw_text:
        return text
    try:
        head, rest1 = text.split("===CORRECTED===", 1)
        corrected_block, rest2 = rest1.split("===CHANGES===", 1)
        changes_block, tail = rest2.split("===END===", 1)
    except ValueError:
        return text

    new_corrected = corrected_block
    kept: list[str] = []
    dropped_count = 0
    reverted_in_corrected = 0  # v1.7.1: число compound-revert'ов в CORRECTED
    for raw_line in changes_block.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            kept.append(line)
            continue
        m = _CHANGE_PAIR_RE.search(line)
        if m:
            quote_before = m.group(1).strip()
            quote_after = m.group(2).strip()
            # v1.7: single-word path — если вся цитата это одно
            # слово, и оно подменено падежной формой — дропаем
            # пункт целиком, откатываем в CORRECTED.
            if (
                _morph_filter.is_hallucinated_case_change(
                    quote_before, quote_after, raw_text
                )
                and quote_after in new_corrected
            ):
                new_corrected = new_corrected.replace(quote_after, quote_before)
                logger.info(
                    "Дроп падежной «улучшалки»: «%s» → «%s» "
                    "(одна лемма, разный падеж, нет управляющего предлога; откат в CORRECTED)",
                    quote_before, quote_after,
                )
                dropped_count += 1
                continue
            # v1.7.1: compound path — модель часто упаковывает несколько
            # правок в одну цитату («повлекших риски причинения ущерба
            # Подразделения» → «повлёкших риски причинения ущерба
            # Подразделению»). Single-word check выше пропускает такие
            # кейсы, но внутри компаунда может прятаться галлюцинация
            # вроде «Подразделения → Подразделению». Откатываем такие
            # отдельные слова в CORRECTED. Если ВЕСЬ компаунд состоит
            # из таких подмен (плюс ё-различий, обрабатываемых
            # eyo-undo) — дропаем пункт целиком, иначе оставляем
            # пункт (там есть реальная правка).
            pairs = _morph_filter.find_hallucinated_pairs_in_compound(
                quote_before, quote_after, raw_text
            )
            if pairs:
                for bw, aw in pairs:
                    if aw in new_corrected:
                        new_corrected = new_corrected.replace(aw, bw)
                        reverted_in_corrected += 1
                        logger.info(
                            "Дроп падежной «улучшалки» (compound): «%s» → «%s» "
                            "(внутри компаунда «%s» → «%s»; откат в CORRECTED)",
                            bw, aw, quote_before[:60], quote_after[:60],
                        )
                if _morph_filter.is_compound_fully_hallucinated(
                    quote_before, quote_after, raw_text
                ):
                    dropped_count += 1
                    continue
                # mixed: реальная правка есть, оставляем пункт CHANGES,
                # но `reverted_in_corrected` гарантирует, что обновлённый
                # CORRECTED дойдёт до выхода (см. условие ниже).
        kept.append(line)

    if dropped_count == 0 and reverted_in_corrected == 0:
        return text

    non_empty = [ln for ln in kept if re.search(r"\w", ln)]
    has_real_item = any(re.match(r"\s*\d+\.\s*\S", ln) for ln in non_empty)
    if not has_real_item:
        kept = ["", "1. Ошибок не найдено. Текст соответствует нормам.", ""]

    if dropped_count > 0:
        logger.info(
            "Отфильтровано %d пункт(ов) падежных «улучшений» (pymorphy3 morph-filter)",
            dropped_count,
        )
    if reverted_in_corrected > 0:
        logger.info(
            "Откат %d compound-словоподмен в CORRECTED (pymorphy3 morph-filter)",
            reverted_in_corrected,
        )

    new_changes = "\n".join(kept).rstrip() + "\n"
    return (
        f"{head}===CORRECTED==={new_corrected}"
        f"===CHANGES===\n{new_changes.lstrip()}===END==={tail}"
    )


def _expand_word_context(s: str, lo: int, hi: int) -> tuple[int, int]:
    """Расширяет [lo, hi) до границ слов с прихватом одного соседнего слова
    с каждой стороны. Используется для генерации читаемых «было»/«стало»
    в `_rebuild_changes_from_diff` — голый character-level diff даёт обрывки
    типа «,» вместо «задач, по», что бесполезно пользователю и неуникально
    для клиентского InStr."""
    # Сворачиваем влево до начала текущего слова
    while lo > 0 and not s[lo - 1].isspace():
        lo -= 1
    # Прихватываем одно предыдущее слово (через пробел)
    if lo > 0:
        while lo > 0 and s[lo - 1].isspace():
            lo -= 1
        while lo > 0 and not s[lo - 1].isspace():
            lo -= 1
    # Сворачиваем вправо до конца текущего слова
    while hi < len(s) and not s[hi].isspace():
        hi += 1
    # Прихватываем одно следующее слово
    if hi < len(s):
        while hi < len(s) and s[hi].isspace():
            hi += 1
        while hi < len(s) and not s[hi].isspace():
            hi += 1
    return lo, hi


def _rebuild_changes_from_diff(raw_text: str, corrected_text: str) -> list[str]:
    """Восстанавливает пункты `===CHANGES===` из посимвольного diff между
    исходным текстом пользователя и `===CORRECTED===` из ответа модели.

    Применяется когда модель (yandex-corrector и подобные) выдаёт
    правильный исправленный текст, но сочиняет неправдоподобный отчёт
    в `===CHANGES===` (например, единственный пункт про несуществующее
    в тексте слово). После `_drop_changes_not_in_text` все её пункты
    выкидываются, и без реконструкции пользователь видит «Ошибок не
    найдено», хотя реально в CORRECTED исправления есть.

    Возвращает список строк-пунктов (без нумерации, без обёртки) или
    пустой список, если raw_text == corrected_text.

    Каждый пункт: «фрагмент_исходника_с_контекстом» → «фрагмент_исправления_с_контекстом»."""
    if not raw_text or not corrected_text or raw_text == corrected_text:
        return []

    sm = difflib.SequenceMatcher(None, raw_text, corrected_text, autojunk=False)
    entries: list[str] = []
    seen_before: set[str] = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        a, b = _expand_word_context(raw_text, i1, i2)
        c, d = _expand_word_context(corrected_text, j1, j2)
        before_part = raw_text[a:b].strip()
        after_part = corrected_text[c:d].strip()
        if not before_part or before_part == after_part:
            continue
        # «было» должно быть substring исходника — это инвариант (мы из него
        # же и вырезаем), но проверим на всякий случай: пробелы внутри/вокруг
        # могли уехать при .strip().
        if before_part not in raw_text:
            continue
        # Дедупликация: если несколько diff-опкодов попали в одно
        # расширенное окно, эмитим только первый.
        if before_part in seen_before:
            continue
        seen_before.add(before_part)
        entries.append(
            f"«{before_part}» → «{after_part}» | автоправка по diff "
            f"(модель не указала точную причину)"
        )
    return entries


def _has_real_change_items(text: str) -> bool:
    """Возвращает True, если в `===CHANGES===` есть хотя бы один
    содержательный пункт (не «Ошибок не найдено», не идемпотентный,
    не комментарий)."""
    if "===CHANGES===" not in text or "===END===" not in text:
        return False
    try:
        _, rest = text.split("===CHANGES===", 1)
        block, _ = rest.split("===END===", 1)
    except ValueError:
        return False
    for line in block.splitlines():
        if not line.strip():
            continue
        if "Ошибок не найдено" in line:
            continue
        m = _CHANGE_PAIR_RE.search(line)
        if m and m.group(1).strip() and m.group(1).strip() != m.group(2).strip():
            return True
    return False


def _had_any_change_pairs(text: str) -> bool:
    """Был ли в `===CHANGES===` хотя бы один пункт «X» → «Y» ДО фильтрации.
    Используется чтобы отличить «модель отдала пары, мы их выкинули как
    галлюцинации» (надо реконструировать) от «модель не выдала вообще
    никакого формата» (реконструировать нельзя — CORRECTED это garbage)."""
    if "===CHANGES===" not in text:
        return False
    try:
        _, rest = text.split("===CHANGES===", 1)
    except ValueError:
        return False
    block = rest.split("===END===", 1)[0] if "===END===" in rest else rest
    for line in block.splitlines():
        if _CHANGE_PAIR_RE.search(line):
            return True
    return False


def _extract_corrected_body(text: str) -> str:
    """Возвращает содержимое блока `===CORRECTED===` без переносов в начале/конце.
    Возвращает пустую строку, если маркеры отсутствуют или порядок нарушен."""
    if "===CORRECTED===" not in text or "===CHANGES===" not in text:
        return ""
    try:
        _, after = text.split("===CORRECTED===", 1)
        body, _ = after.split("===CHANGES===", 1)
    except ValueError:
        return ""
    return body.strip()


def _replace_changes_block(text: str, entries: list[str]) -> str:
    """Заменяет содержимое блока `===CHANGES===` на пронумерованный список
    `entries`. Если entries пусто — оставляет «Ошибок не найдено»."""
    if "===CHANGES===" not in text or "===END===" not in text:
        return text
    try:
        before, rest = text.split("===CHANGES===", 1)
        _, tail = rest.split("===END===", 1)
    except ValueError:
        return text
    if entries:
        new_block = "\n".join(f"{i}. {e}" for i, e in enumerate(entries, start=1))
    else:
        new_block = "1. Ошибок не найдено. Текст соответствует нормам."
    return f"{before}===CHANGES===\n{new_block}\n===END==={tail}"


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
