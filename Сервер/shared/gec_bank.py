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

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .rag_store import Embedder, HashingEmbedder  # noqa: F401 — для потребителей

_log = logging.getLogger("ai_suggester.gec_bank")

# Кавычки, которые сервер использует как ВНЕШНИЕ ограничители в блоке
# `===CHANGES===` (см. `_QUOTE_CHARS` в `Сервер/local/main.py`). Если
# пара «неправильно → правильно» сама содержит любой из этих символов,
# то few-shot assistant-сообщение получит вложенные кавычки вида
# `«текст с «внутренней» кавычкой»`, и серверный регэксп
# `_CHANGE_PAIR_RE` обрежет цитату до первой внутренней кавычки. Модель,
# обученная по этим примерам, воспроизведёт паттерн → `_drop_changes_not_in_text`
# не найдёт цитату в `raw_text` и молча выкинет валидную правку. Такие пары
# исключаются из банка на этапе загрузки (см. load_jsonl).
_OUTER_QUOTE_CHARS = "«»\"\u201c\u201d\u2018\u2019\u201a\u201b\u201e"


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
    def build_index(self) -> None:
        """Считает эмбеддинги для всех пар и сохраняет в RAM.

        Эмбеддим `wrong`-поле: пользовательский текст тоже «неправильный»,
        поэтому максимально близкий по смыслу пример будет иметь похожую
        структуру ошибки. Для nomic-embed-text это даёт интуитивно
        правильное поведение (похожие контексты → похожие правила).
        """
        if not self._entries:
            self._indexed_count = 0
            return
        texts = [e.pair.wrong for e in self._entries]
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

    # --- поиск ---------------------------------------------------------
    def search(self, query: str, top_k: int = 3) -> List[Tuple[float, GecPair]]:
        """Возвращает top-k пар, наиболее похожих на query, по cosine.

        Возвращает список `(score, pair)` в порядке убывания score.
        Если банк пуст или не проиндексирован — пустой список.
        """
        if not self._entries or self._indexed_count == 0:
            return []
        qvec = self.embedder.embed([query])[0]
        qnorm = math.sqrt(sum(v * v for v in qvec)) or 1.0
        scored: List[Tuple[float, GecPair]] = []
        for entry in self._entries:
            if not entry.vec or len(entry.vec) != len(qvec):
                continue  # эмбеддер сменился или не проиндексирована пара
            dot = sum(a * b for a, b in zip(qvec, entry.vec))
            scored.append((dot / (qnorm * entry.norm), entry.pair))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

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
