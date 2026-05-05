"""
Банк эталонных пар «неправильно → правильно» для few-shot retrieval.

Идея:
  Вместо того, чтобы надеяться, что LLM «сама вспомнит» правила русской
  грамматики и пунктуации, мы подмешиваем в каждый запрос 3-5 самых похожих
  примеров из заранее собранного банка. Это **retrieval-augmented few-shot**
  прямо как в paper Sorokin & Nasyrova (BEA 2025) — на LORuGEC это
  повышает F0.5 Russian GEC с ~55 % (zero-shot) до ~83 % (5-shot с
  GECToR retrieval).

Архитектура:
  * Банк хранится в JSONL: один пример на строку, формат GecPair.
  * Эмбеддер — любой, соответствующий протоколу `shared.rag_store.Embedder`
    (по умолчанию OllamaEmbedder с nomic-embed-text или HashingEmbedder для
    офлайн-режима).
  * Индекс — в RAM, numpy-cosine. Для 288 пар это <50 мс поиска на Broadwell,
    FAISS не нужен до ~10 К пар.

Источник seed-банка:
  LORuGEC (Sorokin & Nasyrova 2025, BEA @ ACL 2025) —
  https://github.com/ReginaNasyrova/LORuGEC. 48 правил русской грамматики,
  960 rule-annotated пар. Цитирование обязательно.

Расширение под ведомственные документы:
  Загрузка поддерживает несколько JSONL через `load_jsonl(*paths)`. Когда
  у вас будет корпус из реальных пар «проект документа → утверждённая
  редакция», сохраните их в отдельный файл (например
  `data/gec_bank_agency.jsonl` того же формата) и добавьте путь в
  `GEC_BANK_FILES` в `.env`. Пересборка индекса — `GecBank.build_index()`,
  это занимает секунды для сотен пар.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .rag_store import Embedder, HashingEmbedder  # noqa: F401 — для потребителей

_log = logging.getLogger("ai_suggester.gec_bank")

# Кавычки, которые сервер использует как ВНЕШНИЕ ограничители в блоке
# `===CHANGES===` (см. `_QUOTE_CHARS` в `server/local/main.py`). Если
# пара «неправильно → правильно» сама содержит любой из этих символов,
# то few-shot assistant-сообщение получит вложенные кавычки вида
# `«текст с «внутренней» кавычкой»`, и серверный регэксп
# `_CHANGE_PAIR_RE` обрежет цитату до первой внутренней кавычки. Модель,
# обученная по этим примерам, воспроизведёт паттерн → `_drop_changes_not_in_text`
# не найдёт цитату в `raw_text` и молча выкинет валидную правку. Такие пары
# исключаются из банка на этапе загрузки (см. load_jsonl).
_OUTER_QUOTE_CHARS = "«»\"\u201c\u201d\u2018\u2019\u201a\u201b\u201e"

# Токенизация для BM25 (sparse retrieval). Берём слова из кириллицы +
# латиницы + цифр; нормализуем регистр и снимаем разницу ё/е (в большинстве
# официальных документов е и ё взаимозаменяемы, и BM25 не должен их различать).
_BM25_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+")


def _tokenize_ru(text: str) -> List[str]:
    """Лёгкий токенизатор для BM25. Без лемматизации: ловим словоформы
    напрямую, потому что грамматическая ошибка как раз в словоформе
    («стоимостей» vs «стоимости», «выполненной» vs «выполненных»).
    Если стеммировать — потеряем главный сигнал.
    """
    return _BM25_TOKEN_RE.findall(text.lower().replace("ё", "е"))


class _BM25Index:
    """Чистый Python BM25Okapi над статичной коллекцией документов.

    Используется как sparse-половина гибридного retrieval (см. v1.6.7).
    Для 894 пар банка строится за ~10 мс при старте, занимает ~2-5 МБ RAM,
    кэш на диск не нужен. Параметры k1=1.5, b=0.75 — стандартные значения
    Okapi BM25 (Robertson 2009), хорошо работают на коротких документах.
    """

    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = len(docs)
        self.doc_lens = [len(d) for d in docs]
        self.avgdl = (sum(self.doc_lens) / self.N) if self.N else 0.0
        self.tfs: List[Dict[str, int]] = []
        df: Dict[str, int] = {}
        for d in docs:
            tf: Dict[str, int] = {}
            for tok in d:
                tf[tok] = tf.get(tok, 0) + 1
            self.tfs.append(tf)
            for tok in tf.keys():
                df[tok] = df.get(tok, 0) + 1
        # Okapi IDF со сглаживанием +0.5 (стандарт)
        self.idf: Dict[str, float] = {
            tok: math.log((self.N - dfi + 0.5) / (dfi + 0.5) + 1.0)
            for tok, dfi in df.items()
        }

    def score(self, query: List[str]) -> List[float]:
        """Возвращает BM25-score для каждого документа (по индексу)."""
        scores = [0.0] * self.N
        if self.avgdl <= 0:
            return scores
        for tok in query:
            idf = self.idf.get(tok)
            if idf is None:
                continue
            for i, tf in enumerate(self.tfs):
                f = tf.get(tok, 0)
                if f == 0:
                    continue
                dl = self.doc_lens[i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * f * (self.k1 + 1) / denom
        return scores


@dataclass
class GecPair:
    """Один пример в банке.

    Attributes:
        wrong: Фрагмент текста с ошибкой. По нему считается эмбеддинг и
            идёт поиск ближайших соседей к пользовательскому тексту.
        right: Корректная версия того же фрагмента.
        rule: Короткое имя правила (например, «Запятая при обособленном
            определении»). Показывается в логе и может подмешиваться как
            подпись в few-shot CHANGES.
        definition: Полная формулировка правила (опционально). Помогает
            LLM воспроизвести причину в CHANGES при few-shot.
        section: Секция грамматики (Punctuation / Spelling / ...) —
            опционально, для диагностики.
    """

    wrong: str
    right: str
    rule: str = ""
    definition: str = ""
    section: str = ""


@dataclass
class _Indexed:
    pair: GecPair
    vec: List[float] = field(default_factory=list)
    norm: float = 1.0


class GecBank:
    """Банк GEC-пар с эмбеддингами и поиском ближайших соседей."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._entries: List[_Indexed] = []
        self._indexed_count = 0
        self._bm25: Optional[_BM25Index] = None

    # --- загрузка ------------------------------------------------------
    def load_jsonl(self, *paths: Path | str) -> int:
        """Загружает пары из одного или нескольких JSONL-файлов.

        Поля `wrong` и `right` обязательны; отсутствующие строки
        игнорируются с warning'ом. Пары, содержащие в `wrong` или `right`
        любой из `_OUTER_QUOTE_CHARS`, пропускаются и счётчик таких
        skipped-пар логируется — они создают вложенные кавычки в
        few-shot CHANGES и ломают `_CHANGE_PAIR_RE` на стороне сервера
        (`_drop_changes_not_in_text` молча выкидывает валидные правки).

        Возвращает число успешно загруженных пар.
        """
        loaded = 0
        for p_raw in paths:
            p = Path(p_raw)
            if not p.exists():
                _log.warning("GEC bank файл не найден: %s", p)
                continue
            skipped_nested = 0
            with p.open("r", encoding="utf-8") as f:
                for lineno, raw_line in enumerate(f, 1):
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        _log.warning("GEC bank %s:%d: невалидный JSON — %s", p, lineno, e)
                        continue
                    wrong = (data.get("wrong") or "").strip()
                    right = (data.get("right") or "").strip()
                    if not wrong or not right or wrong == right:
                        continue
                    if any(c in wrong or c in right for c in _OUTER_QUOTE_CHARS):
                        skipped_nested += 1
                        continue
                    self._entries.append(_Indexed(pair=GecPair(
                        wrong=wrong,
                        right=right,
                        rule=(data.get("rule") or "").strip(),
                        definition=(data.get("definition") or "").strip(),
                        section=(data.get("section") or "").strip(),
                    )))
                    loaded += 1
            if skipped_nested:
                _log.info(
                    "GEC bank: %s — пропущено %d пар с вложенными кавычками "
                    "(защита от обрезки _CHANGE_PAIR_RE)", p, skipped_nested,
                )
            _log.info("GEC bank: загружен %s (всего пар: %d)", p, len(self._entries))
        return loaded

    def add_pair(self, pair: GecPair) -> None:
        self._entries.append(_Indexed(pair=pair))

    def __len__(self) -> int:
        return len(self._entries)

    # --- индексация ----------------------------------------------------
    def _bank_fingerprint(self) -> str:
        """SHA-256 по `(wrong, right)`-парам банка + имени эмбеддера.

        Используется как ключ кэша. Изменили банк или эмбеддер — ключ меняется,
        кэш переиндексируется. Rule/definition в хэше не учитываем: эмбеддинг
        берём только с `wrong`, поэтому перефразировка правила не обнуляет кэш.
        """
        h = hashlib.sha256()
        h.update(self.embedder.name.encode("utf-8"))
        h.update(b"\n")
        for e in self._entries:
            h.update(e.pair.wrong.encode("utf-8"))
            h.update(b"\t")
            h.update(e.pair.right.encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()

    def build_index(self, cache_path: Path | str | None = None) -> None:
        """Считает эмбеддинги для всех пар и сохраняет в RAM.

        Эмбеддим `wrong`-поле: пользовательский текст тоже «неправильный»,
        поэтому максимально близкий по смыслу пример будет иметь похожую
        структуру ошибки. Для nomic-embed-text это даёт интуитивно
        правильное поведение (похожие контексты → похожие правила).

        Если передан `cache_path` — перед обсчётом пытаемся загрузить
        ранее сохранённые векторы (pickle) с совпадающим fingerprint банка
        и эмбеддера. Это критично для старта на CPU Broadwell: без кэша
        278 последовательных `/api/embeddings`-запросов к Ollama
        (nomic-embed-text, ~8 с на запрос) занимают ~37 минут.

        При переиндексации — сохраняем кэш на диск под тем же `cache_path`.
        """
        if not self._entries:
            self._indexed_count = 0
            return

        fp = self._bank_fingerprint()
        cache = Path(cache_path) if cache_path else None

        # Попытка загрузить кэш
        if cache is not None and cache.exists():
            try:
                with cache.open("rb") as f:
                    payload = pickle.load(f)
                if payload.get("fingerprint") == fp:
                    vecs = payload["vectors"]
                    if len(vecs) == len(self._entries):
                        for entry, vec in zip(self._entries, vecs):
                            entry.vec = list(vec)
                            entry.norm = (
                                math.sqrt(sum(v * v for v in vec)) or 1.0
                            )
                        self._indexed_count = len(self._entries)
                        _log.info(
                            "GEC bank: загружен кэш индекса %s (%d пар, dim=%d, embedder=%s)",
                            cache, self._indexed_count,
                            len(vecs[0]) if vecs else 0, self.embedder.name,
                        )
                        return
                _log.info(
                    "GEC bank: fingerprint кэша %s устарел — переиндексация", cache,
                )
            except Exception as e:
                _log.warning("GEC bank: кэш %s повреждён (%s) — переиндексация", cache, e)

        # Обсчёт индекса
        texts = [e.pair.wrong for e in self._entries]
        _log.info(
            "GEC bank: считаю эмбеддинги для %d пар через %s "
            "(на CPU может занять несколько минут)…",
            len(texts), self.embedder.name,
        )
        vecs = self.embedder.embed(texts)
        for entry, vec in zip(self._entries, vecs):
            entry.vec = list(vec)
            entry.norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        self._indexed_count = len(self._entries)
        _log.info(
            "GEC bank: проиндексировано %d пар, embedder=%s, dim=%d",
            self._indexed_count, self.embedder.name,
            len(self._entries[0].vec) if self._entries else 0,
        )

        # Сохранение кэша
        if cache is not None:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache.with_suffix(cache.suffix + ".tmp")
                with tmp.open("wb") as f:
                    pickle.dump(
                        {
                            "fingerprint": fp,
                            "vectors": [list(e.vec) for e in self._entries],
                            "embedder": self.embedder.name,
                        },
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                tmp.replace(cache)
                _log.info("GEC bank: сохранён кэш индекса %s", cache)
            except Exception as e:
                _log.warning(
                    "GEC bank: не удалось сохранить кэш %s (%s) — индекс только в RAM",
                    cache, e,
                )

        # Sparse-половина гибридного retrieval. Кэш не нужен — для 894 пар
        # построение BM25 занимает <20 мс, не критично к старту.
        self.build_bm25_index()

    def build_bm25_index(self) -> None:
        """Строит BM25-индекс для sparse retrieval (используется в search_hybrid).

        Эмбеддит токенизированные «wrong»-поля. Запускается автоматически
        в конце build_index(), но можно дёрнуть вручную (например, в тестах).
        """
        if not self._entries:
            self._bm25 = None
            return
        docs = [_tokenize_ru(e.pair.wrong) for e in self._entries]
        self._bm25 = _BM25Index(docs)
        _log.info(
            "GEC bank: BM25 sparse-индекс собран (%d пар, словарь %d уникальных токенов)",
            len(self._entries), len(self._bm25.idf),
        )

    # --- поиск ---------------------------------------------------------
    def search(self, query: str, top_k: int = 3) -> List[Tuple[float, GecPair]]:
        """Возвращает top-k пар, наиболее похожих на query, по cosine
        (dense retrieval). Backward-совместимый API.

        Возвращает список `(score, pair)` в порядке убывания score.
        Если банк пуст или не проиндексирован — пустой список.
        """
        ranked = self._dense_rank(query)
        return [(s, self._entries[i].pair) for s, i in ranked[:top_k]]

    def search_sparse(self, query: str, top_k: int = 3) -> List[Tuple[float, GecPair]]:
        """BM25-поиск (sparse). Ловит точные словоформы, в т.ч. редкие
        падежные формы — для GEC это часто важнее семантики (ошибка
        и есть в словоформе).
        """
        ranked = self._sparse_rank(query)
        return [(s, self._entries[i].pair) for s, i in ranked[:top_k]]

    def search_hybrid(
        self,
        query: str,
        top_k: int = 3,
        pool: Optional[int] = None,
        rrf_k: int = 60,
    ) -> List[Tuple[float, GecPair]]:
        """Гибридный retrieval: RRF-фьюжн dense (cosine) + sparse (BM25).

        Reciprocal Rank Fusion (Cormack et al. 2009): из каждого ранжирования
        берём 1/(k+rank), суммируем. k=60 — стандартное значение, гасит
        нерелевантные хвосты обоих списков.

        Зачем: dense на длинных входах часто промахивается (Sorokin & Nasyrova
        BEA 2025: «similarity between input texts does not necessarily
        correspond to similar grammatical error patterns»). BM25 на тех же
        входах ловит редкие словоформы — а ошибка как раз в словоформе.
        Фьюжн объединяет два сигнала без подкрутки шкал.

        Args:
            query: текст запроса.
            top_k: сколько пар вернуть финально.
            pool: размер кандидатного пула из каждого ранжирования (default
                max(top_k*4, 10)). Больше pool — выше recall, но дороже фьюжн.
            rrf_k: константа RRF (default 60).

        Returns: список `(rrf_score, pair)` в порядке убывания.
        """
        if not self._entries or self._indexed_count == 0:
            return []
        eff_pool = pool if pool is not None else max(top_k * 4, 10)
        dense_ranked = self._dense_rank(query)[:eff_pool]
        sparse_ranked = self._sparse_rank(query)[:eff_pool] if self._bm25 else []

        # 1-based ранг каждого entry-индекса в каждом ранжировании.
        dense_rank: Dict[int, int] = {idx: pos + 1 for pos, (_, idx) in enumerate(dense_ranked)}
        sparse_rank: Dict[int, int] = {idx: pos + 1 for pos, (_, idx) in enumerate(sparse_ranked)}
        candidates = set(dense_rank) | set(sparse_rank)

        fused: List[Tuple[float, int]] = []
        for idx in sorted(candidates):
            s = 0.0
            if idx in dense_rank:
                s += 1.0 / (rrf_k + dense_rank[idx])
            if idx in sparse_rank:
                s += 1.0 / (rrf_k + sparse_rank[idx])
            fused.append((s, idx))
        # При равном RRF-скоре тай-брейк по индексу пары (меньший индекс
        # выше) — иначе порядок внутри set(candidates) недетерминирован
        # между процессами и одинаковые запросы возвращают разные пары
        # между рестартами сервера. Реальный кейс: тот же КС-2 текст в
        # v1.6.8 первого/второго прогона дал разные top-1 (`Запятая перед
        # союзом "как"` vs другие) — это симптом этого недетерминизма.
        fused.sort(key=lambda x: (-x[0], x[1]))
        return [(s, self._entries[i].pair) for s, i in fused[:top_k]]

    # --- ранжирование (внутренние) ------------------------------------
    def _dense_rank(self, query: str) -> List[Tuple[float, int]]:
        """Cosine-ранжирование всех проиндексированных entry; (score, idx) desc."""
        if not self._entries or self._indexed_count == 0:
            return []
        qvec = self.embedder.embed([query])[0]
        qnorm = math.sqrt(sum(v * v for v in qvec)) or 1.0
        scored: List[Tuple[float, int]] = []
        for i, entry in enumerate(self._entries):
            if not entry.vec or len(entry.vec) != len(qvec):
                continue  # эмбеддер сменился или пара не проиндексирована
            dot = sum(a * b for a, b in zip(qvec, entry.vec))
            scored.append((dot / (qnorm * entry.norm), i))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def _sparse_rank(self, query: str) -> List[Tuple[float, int]]:
        """BM25-ранжирование всех entry; (score, idx) desc. Без BM25-индекса — пусто."""
        if self._bm25 is None or not self._entries:
            return []
        qtoks = _tokenize_ru(query)
        scores = self._bm25.score(qtoks)
        scored = [(s, i) for i, s in enumerate(scores) if s > 0.0]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    # --- API для отладки -----------------------------------------------
    def stats(self) -> dict:
        rules: dict[str, int] = {}
        for e in self._entries:
            rules[e.pair.rule] = rules.get(e.pair.rule, 0) + 1
        return {
            "total_pairs": len(self._entries),
            "indexed_pairs": self._indexed_count,
            "embedder": self.embedder.name,
            "rules": len(rules),
            "top_rules": sorted(rules.items(), key=lambda x: -x[1])[:5],
            "bm25_terms": (len(self._bm25.idf) if self._bm25 else 0),
        }


# ── Форматирование few-shot-сообщений ───────────────────────────────
def format_example_as_messages(pair: GecPair) -> List[dict]:
    """Преобразует одну пару в шаблон `user → assistant`.

    Assistant-сообщение полностью повторяет формат ответа сервера
    (===CORRECTED=== / ===CHANGES=== / ===END===). Это обучает модель
    держать формат: 3-5 таких пар в истории → модель копирует шаблон
    вместо того, чтобы изобретать свой.
    """
    user = f"ТЕКСТ ДЛЯ ПРОВЕРКИ:\n{pair.wrong}"
    cause = pair.rule or "пунктуация"
    assistant = (
        "===CORRECTED===\n"
        f"{pair.right}\n"
        "===CHANGES===\n"
        f"1. «{pair.wrong}» → «{pair.right}» | {cause}\n"
        "===END==="
    )
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def build_few_shot_messages(
    system_prompt: str,
    user_text: str,
    examples: Iterable[GecPair],
    extra_system: Optional[str] = None,
) -> List[dict]:
    """Собирает полный список `messages` для call_ollama.

    Порядок:
        1. system (основной промпт с правилами формата)
        2. опциональный system с RAG-контекстом
        3. N пар (user, assistant) — few-shot примеры
        4. финальный user с реальным текстом пользователя

    Некоторые Ollama-шаблоны (например Qwen-ChatML) корректно видят
    несколько system-сообщений. Если модель это не поддерживает,
    extra_system объединится с system_prompt через '\\n\\n'.
    """
    messages: List[dict] = [{"role": "system", "content": system_prompt}]
    if extra_system:
        messages.append({"role": "system", "content": extra_system})
    for pair in examples:
        messages.extend(format_example_as_messages(pair))
    messages.append({"role": "user", "content": user_text})
    return messages
