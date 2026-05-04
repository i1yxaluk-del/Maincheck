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
MODEL_NAME = os.getenv("MODEL_NAME", "t-tech/T-lite-it-2.1:q4_K_M")
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
    s.strip() for s in os.getenv("GEC_BANK_FILES", "../shared/gec_seed/gec_bank.jsonl").split(",")
    if s.strip()
]
GEC_TOP_K = int(os.getenv("GEC_TOP_K", "3"))

logger = setup_logger("ai_suggester.local")
audit = AuditStore()

_rag_store = None
_rag_embedder = None
_gec_bank = None

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

        # Переиспользуем RAG-эмбеддер, если он уже инициализирован (экономим RAM
        # и сетевые hops). Иначе создаём собственный Ollama-эмбеддер на той же
        # модели nomic-embed-text. Если Ollama недоступен — HashingEmbedder как
        # резерв (лексическое пересечение вместо семантического, качество ниже,
        # но не ломается).
        if _rag_embedder is not None:
            _gec_embedder = _rag_embedder
        else:
            try:
                _gec_embedder = OllamaEmbedder(model=RAG_EMBED_MODEL, base_url=OLLAMA_URL)
                # Проверка доступности: делаем probe-эмбеддинг
                _ = _gec_embedder.embed(["probe"])
            except Exception as exc:
                logger.warning(
                    "Ollama-эмбеддер (%s) недоступен (%s), переключаюсь на HashingEmbedder",
                    RAG_EMBED_MODEL, exc,
                )
                _gec_embedder = HashingEmbedder(dim=1024)
        _gec_bank = GecBank(_gec_embedder)
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
                "Few-shot retrieval включён: %d пар, top_k=%d, embedder=%s",
                n, GEC_TOP_K, _gec_embedder.name,
            )
    except Exception as e:
        logger.warning("Few-shot retrieval не удалось инициализировать: %s", e)
        _gec_bank = None


SYSTEM_PROMPT = """Ты — корректор русского языка для официальных документов. Не рассуждай, сразу выдавай ответ в нужном формате.

ИСПРАВЛЯЙ ТОЛЬКО ЯВНЫЕ ОШИБКИ:
• орфография — опечатки, удвоение/пропуск букв, слитное/раздельное написание;
• управление — «согласно приказу» (не «согласно приказа»), «благодаря решению»;
• согласование — однородные члены в одном падеже и числе с главным словом;
• пунктуация — запятые при однородных членах, обособленных оборотах, придаточных.

НЕ ТРОГАЙ:
• аббревиатуры и сокращения (п/п, вх.№, исх.№, ФСБ, МВД);
• ведомственные термины и профессиональные обороты;
• правильно написанный текст («улучшать стиль» нельзя);
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
    пользователя не ждал 30–90 с на загрузку весов 30B-модели.

    Делает один минимальный chat-запрос с keep_alive — Ollama после
    этого держит модель загруженной OLLAMA_KEEP_ALIVE минут.
    """
    if not OLLAMA_WARMUP:
        return
    logger.info("Прогрев модели %s через Ollama (timeout=%.0fs)…",
                MODEL_NAME, OLLAMA_WARMUP_TIMEOUT)
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_WARMUP_TIMEOUT) as c:
            r = await c.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": "ok\n\n/no_think"}],
                    "stream": False,
                    "think": False,
                    "keep_alive": OLLAMA_KEEP_ALIVE,
                    "options": {"num_ctx": 512, "num_thread": NUM_THREADS},
                },
            )
            r.raise_for_status()
        logger.info("Прогрев OK: модель загружена в RAM, keep_alive=%s",
                    OLLAMA_KEEP_ALIVE)
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
        m = _CHANGE_PAIR_RE.search(line)
        # Сравниваем БЕЗ .lower(): «Приказа» → «приказа» — это валидная
        # орфографическая правка регистра (имя собственное vs нарицательное),
        # такие пункты сохраняем. Идемпотентный пункт — это когда до и после
        # совпадают побуквенно.
        if m and m.group(1).strip() == m.group(2).strip():
            logger.debug("Фильтрую идемпотентный пункт: %s", line.strip())
            continue
        # Пункты с многоточием в цитатах («…» или «...») неприменимы:
        # клиент ищет «было» через InStr в выделении, а сокращённую цитату
        # никогда не найдёт. Лучше скрыть пункт, чем показывать пользователю
        # «не удалось применить (фрагмент не найден)». YandexGPT-5-Lite
        # склонна к таким сокращениям; T-lite — реже. Промпт это запрещает,
        # но оставляем как страховку.
        if m and ("…" in m.group(1) or "..." in m.group(1)
                  or "…" in m.group(2) or "..." in m.group(2)):
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
            "temperature": 0.1,
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
        "rag_enabled": RAG_ENABLED,
        "rag_documents": len(_rag_store.docs) if _rag_store else 0,
        "few_shot_enabled": _gec_bank is not None,
        "few_shot_pairs": len(_gec_bank) if _gec_bank else 0,
        "few_shot_top_k": GEC_TOP_K if _gec_bank else 0,
        "audit": audit.stats(hours=hours),
    })


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
            hits = _gec_bank.search(raw_text, top_k=GEC_TOP_K)
            few_shot_examples = [pair for score, pair in hits]
            if few_shot_examples:
                logger.info(
                    "Few-shot: подмешиваю %d пар: %s",
                    len(few_shot_examples),
                    [p.rule or p.wrong[:40] for p in few_shot_examples],
                )
        except Exception as e:
            logger.warning("Few-shot retrieval провалился (0-shot fallback): %s", e)
            few_shot_examples = []

    if few_shot_examples:
        from shared.gec_bank import build_few_shot_messages  # noqa: E402

        messages = build_few_shot_messages(
            system_prompt=SYSTEM_PROMPT,
            user_text=user_msg,
            examples=few_shot_examples,
        )
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
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
