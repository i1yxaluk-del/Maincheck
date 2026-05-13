"""Natasha-based dependency parser для синтаксического анализа (v1.9).

Цель — заменить ручные skip-rules v1.8.5 (по FP-классу) на универсальную
проверку через дерево зависимостей. Если adj/причастие НЕ в
атрибутивной связи (amod/det/nummod) со следующим существительным —
disagreement-проверка не запускается. Это закрывает целые классы FP
(агенс_в_творительном, объект_предлога, нестандартный порядок слов)
одной правилом вместо 5–7 hardcoded паттернов.

Архитектура:
  * Singleton `get_syntax_parser()` — natasha-пайплайн загружается
    лениво (~5 сек, ~150 МБ RAM) при первом обращении.
  * `parse(text) -> ParsedDoc` — возвращает плоский список токенов
    с UD-relations и char-offsets. Кэш по sha1(text) на 32 ввода.
  * `ParsedDoc.is_attributive_modifier(...)` — главный API для
    morph_detector / morph_filter.

Latency на админ-тексте 40+ слов: ~8 мс warm (бенчмарк 04.05.2026).
ENV: `NATASHA_ENABLED` (default `true` если natasha установлена).

Если natasha не установлена — `available=False`, парсер no-op,
все интегрированные детекторы откатываются на v1.8.5 hardcoded skip-rules
(остаются как fallback). Это позволяет включить/выключить парсер
без правки кода.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass
from typing import Optional

_log = logging.getLogger("ai_suggester.syntax_parser")


# UD-relations, ЯВНО означающие что слово — НЕ-атрибутивный модификатор
# следующего существительного (а агенс, объект предлога, конъюнкт и т.д.).
# При обнаружении такой связи detector/filter ДОЛЖНЫ пропустить
# disagreement-проверку: даже если падежи не совпадают, это валидно.
#
# ВАЖНО: используем только сильные "анти-атрибутивные" сигналы. natasha
# не всегда правильно парсит сложные обороты — её ПОЛОЖИТЕЛЬНЫЙ ответ
# («это amod») может быть ошибочным (например, «Проверочное мероприятия»
# natasha видит как PROPN nsubj вместо ADJ amod). Поэтому полагаемся
# только на её НЕГАТИВНЫЕ ответы — "эта связь точно НЕ attributive".
_CLEAR_NON_ATTRIBUTIVE_RELS = frozenset({
    "obl",          # oblique (нон-направленный, часто творительный)
    "obl:agent",    # явный агенс пассива («совершенных сотрудниками»)
    "obj",          # прямой объект глагола/причастия
    "iobj",         # косвенный объект
    "parataxis",    # парcketing-комментарий («... — отметил он»)
    "conj",         # конъюнкт («дом и сад»)
    "ccomp",        # complement-предложение
    "advcl",        # adverbial clause
})

# Relations, при которых модификатор атрибутивно бнут к голове.
# Если mod.rel в этом наборе И mod.head_idx == head_idx, мы ДОВЕРЯЕМ
# этому позитивному сигналу и НЕ пытаемся флагать пару как
# не-attributive (защита от natasha-циклов).
_TRUSTED_ATTRIBUTIVE_RELS = frozenset({
    "amod",        # adjectival modifier
    "det",         # determiner
    "nummod",
    "nummod:gov",
    "appos",
})

# Relations, означающие что слово бнут к предлогу (case) или
# к функциональной голове. Если у токена есть ребёнок с rel=`case`
# и POS=ADP — этот токен находится в prepositional phrase, и НЕ
# модифицирует следующее существительное.
_PREPOSITION_CHILD_REL = "case"
_PREPOSITION_POS = "ADP"


@dataclass(frozen=True)
class SyntaxToken:
    """Один токен парсинга с морфо- и синтакс-метаданными.

    Атрибуты:
        text     — строка слова в исходном виде
        start    — char-offset начала в исходном тексте
        end      — char-offset конца (exclusive)
        pos      — UD POS-тег (NOUN, ADJ, VERB, PRON, ADP, PUNCT...)
        rel      — UD relation к голове (amod, obl:agent, nsubj...)
        head_idx — индекс головы в плоском списке `ParsedDoc.tokens`;
                   -1 если корень предложения
        feats    — morpho-фичи (`Case`, `Number`, `Gender`...) — UD-формат
    """
    text: str
    start: int
    end: int
    pos: str
    rel: str
    head_idx: int
    feats: tuple[tuple[str, str], ...]  # frozen для hashable

    @property
    def feats_dict(self) -> dict[str, str]:
        return dict(self.feats)


@dataclass(frozen=True)
class ParsedDoc:
    """Результат парсинга текста: плоский список токенов через все
    предложения. Используйте `token_at_offset()` чтобы найти токен,
    начинающийся на конкретном char-offset (что соответствует индексу
    в pymorphy-based детекторе).
    """
    tokens: tuple[SyntaxToken, ...]
    _offset_to_index: dict[int, int]

    def token_at_offset(self, offset: int) -> Optional[int]:
        """Возвращает индекс токена, начинающегося на `offset`. None
        если такого нет (например, offset попадает на пробел/пунктуацию)."""
        return self._offset_to_index.get(offset)

    def find_token_index(self, text: str, start_search: int = 0) -> Optional[int]:
        """Ищет первый токен с `text` начиная с индекса `start_search`."""
        for i in range(start_search, len(self.tokens)):
            if self.tokens[i].text == text:
                return i
        return None

    def is_clearly_non_attributive(self, mod_idx: int, head_idx: int) -> bool:
        """True если пара (mod, head) ЯВНО НЕ atributive-связь — т.е.
        синтаксис указывает что эти два слова не должны согласовываться.

        Главное правило v1.9. Используется в morph_detector и morph_filter
        для пропуска FP-классов:
          * «совершенных сотрудниками» — obl:agent → True (skip)
          * «при этом количество» — этом.case-child=при → True (skip)
          * «1, 2, 3» — conj → True (skip)

        Возвращает False в ВСЕХ остальных случаях, включая ситуации
        когда natasha мис-парсит структуру. Это безопасное поведение:
        false-negative parser приведёт лишь к падению на v1.8.5
        hardcoded skip-rules + pymorphy3-морфочеку (как раньше).

        ВАЖНО: НЕ полагаемся на natasha-positive-signal `amod` — он
        может быть ложным (например, «Проверочное мероприятия» natasha
        парсит «Проверочное» как PROPN nsubj, теряя amod-связь).
        Полагаемся только на негативные сигналы.
        """
        if mod_idx < 0 or mod_idx >= len(self.tokens):
            return False
        if head_idx < 0 or head_idx >= len(self.tokens):
            return False
        mod = self.tokens[mod_idx]
        head = self.tokens[head_idx]
        # ЗАЩИТА от natasha-циклов: если mod сам тэгирован amod/det/nummod
        # к голове — ДОВЕРЯЕМ позитивному сигналу parser'а и НЕ флагаем
        # пару как non-attributive (даже если head обратно бнут к mod
        # через obj/obl). Закрывает false-negative для нормальных
        # adj-noun словосочетаний типа «Проверочное мероприятия» где
        # natasha выдаёт (token[0].rel=amod, head=token[1]) И
        # (token[1].rel=obj, head=token[0]) — цикл.
        if mod.head_idx == head_idx and mod.rel in _TRUSTED_ATTRIBUTIVE_RELS:
            return False
        # Сигнал 1: модификатор находится в prep-phrase (имеет case-child
        # предлог). Пример: «при этом количество» — у «этом» есть case-child
        # «при». «этом» не модифицирует «количество», даже если natasha
        # цепляет head_idx → kolichestvo с rel=nmod.
        if self._has_preposition_child(mod_idx):
            return True
        # Сигнал 2: head-токен бнут к mod через не-атрибутивный rel
        # (obl:agent, obj, parataxis, ...). Закрывает «проводимых
        # подразделениями»: detector передаёт (mod=проводимых,
        # head=подразделениями), и natasha выдаёт
        # «подразделениями».head=проводимых rel=obl:agent.
        if head.head_idx == mod_idx and head.rel in _CLEAR_NON_ATTRIBUTIVE_RELS:
            return True
        return False

    def is_attributive_modifier(self, mod_idx: int, head_idx: int) -> bool:
        """DEPRECATED (v1.9-rc): используйте `is_clearly_non_attributive`.

        Этот метод оставлен для совместимости с тестами/кодом, который
        ожидает positive-сигнал. НЕ полагайтесь на него для пропуска
        проверок (см. docstring `is_clearly_non_attributive`).
        """
        return not self.is_clearly_non_attributive(mod_idx, head_idx)

    def _has_preposition_child(self, token_idx: int) -> bool:
        """True если у токена есть ребёнок-предлог (rel=case, POS=ADP)."""
        if token_idx < 0 or token_idx >= len(self.tokens):
            return False
        for i, t in enumerate(self.tokens):
            if i == token_idx:
                continue
            if t.head_idx == token_idx and t.rel == _PREPOSITION_CHILD_REL \
                    and t.pos == _PREPOSITION_POS:
                return True
        return False

    def head_token_of(self, token_idx: int) -> Optional[int]:
        """Возвращает индекс головы (или None если root/out-of-range)."""
        if token_idx < 0 or token_idx >= len(self.tokens):
            return None
        h = self.tokens[token_idx].head_idx
        return h if h >= 0 else None


class SyntaxParser:
    """Singleton-обёртка над natasha-пайплайном с lazy-init и кэшем.

    Использование:
        parser = get_syntax_parser()
        if parser.available:
            doc = parser.parse(raw_text)
            if doc and doc.is_attributive_modifier(adj_idx, noun_idx):
                # проверяем disagreement через pymorphy3
                ...
    """

    _CACHE_SIZE = 32

    def __init__(self) -> None:
        self._segmenter = None
        self._morph_vocab = None
        self._morph_tagger = None
        self._syntax = None
        self._Doc = None
        self._cache: dict[str, ParsedDoc] = {}
        self._cache_order: list[str] = []
        self._init_attempted = False
        self._init_failed = False
        self._lock = threading.Lock()

    def _try_init(self) -> None:
        if self._init_attempted:
            return
        self._init_attempted = True
        try:
            from natasha import (  # type: ignore[import-untyped]
                Segmenter, MorphVocab, NewsEmbedding,
                NewsMorphTagger, NewsSyntaxParser, Doc,
            )
            self._segmenter = Segmenter()
            self._morph_vocab = MorphVocab()
            emb = NewsEmbedding()
            self._morph_tagger = NewsMorphTagger(emb)
            self._syntax = NewsSyntaxParser(emb)
            self._Doc = Doc
            _log.info(
                "SyntaxParser: natasha pipeline загружен "
                "(готов парсить дерево зависимостей, ~150 МБ RAM)"
            )
        except Exception as e:
            self._init_failed = True
            _log.warning(
                "SyntaxParser: natasha не загружена (%s) — синтакс-парсер "
                "отключён (откат на v1.8.5 hardcoded skip-rules). "
                "Установите: pip install natasha",
                e,
            )

    @property
    def available(self) -> bool:
        if not self._init_attempted:
            self._try_init()
        return not self._init_failed

    def parse(self, text: str) -> Optional[ParsedDoc]:
        """Парсит текст в `ParsedDoc`. Кэш по sha1(text). None если
        natasha не загружена или парсинг упал.

        Безопасен — любые исключения natasha логируются и не пробрасываются.
        """
        if not self.available:
            return None
        if not text:
            return ParsedDoc(tokens=(), _offset_to_index={})
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        try:
            doc = self._parse_uncached(text)
        except Exception as exc:
            _log.warning(
                "SyntaxParser: parse() упал на text size=%d: %s",
                len(text), exc,
            )
            return None
        with self._lock:
            self._cache[key] = doc
            self._cache_order.append(key)
            while len(self._cache_order) > self._CACHE_SIZE:
                evicted = self._cache_order.pop(0)
                self._cache.pop(evicted, None)
        return doc

    def _parse_uncached(self, text: str) -> ParsedDoc:
        Doc = self._Doc
        assert Doc is not None  # _try_init гарантирует
        doc = Doc(text)
        doc.segment(self._segmenter)
        doc.tag_morph(self._morph_tagger)
        doc.parse_syntax(self._syntax)
        # natasha идентифицирует токены строками "1_1", "1_2" и т.д.
        # Расплющиваем все предложения в плоский список и переводим
        # head_id в индекс плоского списка.
        id_to_flat: dict[str, int] = {}
        tokens_raw = []  # (sent_idx, tok) кортежи
        flat_idx = 0
        for sent in doc.sents:
            for tok in sent.tokens:
                id_to_flat[tok.id] = flat_idx
                tokens_raw.append(tok)
                flat_idx += 1
        tokens: list[SyntaxToken] = []
        for tok in tokens_raw:
            head_idx = id_to_flat.get(tok.head_id, -1)
            feats_raw = tok.feats or {}
            feats_frozen = tuple(sorted(feats_raw.items()))
            tokens.append(SyntaxToken(
                text=tok.text,
                start=int(tok.start),
                end=int(tok.stop),
                pos=tok.pos or "",
                rel=tok.rel or "",
                head_idx=head_idx,
                feats=feats_frozen,
            ))
        offset_map = {t.start: i for i, t in enumerate(tokens)}
        return ParsedDoc(
            tokens=tuple(tokens),
            _offset_to_index=offset_map,
        )


_SINGLETON: Optional[SyntaxParser] = None
_SINGLETON_LOCK = threading.Lock()


def get_syntax_parser() -> SyntaxParser:
    """Возвращает shared-singleton SyntaxParser."""
    global _SINGLETON
    if _SINGLETON is None:
        with _SINGLETON_LOCK:
            if _SINGLETON is None:
                _SINGLETON = SyntaxParser()
    return _SINGLETON


def reset_syntax_parser() -> None:
    """Сбрасывает singleton (для тестов)."""
    global _SINGLETON
    with _SINGLETON_LOCK:
        _SINGLETON = None
