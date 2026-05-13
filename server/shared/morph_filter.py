"""Морфологический фильтр галлюцинированных «улучшений» падежных форм
(v1.7).

Проблема (наблюдалась в v1.6.8/1.6.9 prod-ablation на КС-2 5 мая 2026):
  Модель T-lite-it-2.1 любит «улучшать» уже валид��ые падежные формы:
  «ущерба Подразделения» (родительный падеж от «подразделение») →
  «ущерба Подразделению» (дательный падеж). Формально оба варианта
  правильны, но раз контекст не требует именно дательного, это
  стилистическая правка, не ошибка.

  Char-level фильтр ё (v1.6.9) не закрывает этот класс — там нет ё.
  Текущий `_drop_changes_not_in_text` не помогает: «before» (Подразделения)
  ЕСТЬ в исходном тексте.

Решение:
  pymorphy3 разбирает обе словоформы. Если у них **общая лемма**, **тот
  же грамматический разряд (POS)**, **то же число**, но разные **падежи**
  — и в исходном тексте перед `before` НЕТ предлога, явно требующего
  конкретного падежа — это «улучшение», дроп.

Что НЕ фильтруем (защита от false positive):
  * лексические замены (разные леммы): «принимать» → «принять»;
  * число различается: «выполненной» → «выполненных» (это agreement
    fix причастия с подлежащим — реальная правка);
  * перед `before` стоит case-governing предлог («согласно приказа» →
    «согласно приказу» — реальная ошибка управления);
  * `before` == `after` после нормализации ё/е (это другой фильтр).

Зависимость:
  pymorphy3 + pymorphy3-dicts-ru — pure Python, ~50 МБ словарей, offline.
  Если не установлен — `MorphFilter()` возвращает `available=False`,
  остальной пайплайн работает без морф-фильтра.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from shared.syntax_parser import ParsedDoc

_log = logging.getLogger("ai_suggester.morph_filter")

# v1.9: ENV-флаг включения natasha-парсера в morph_filter. Default `true`.
# Если natasha не установлена — фильтр откатится на v1.8.5 hardcoded
# skip-rules (PRTF+ablt, PREP+местоим).
_NATASHA_ENABLED_ENV = os.getenv("NATASHA_ENABLED", "true").lower() in (
    "1", "true", "yes", "on",
)

# Предлоги/наречия, явно требующие определённого падежа. Если такой
# токен стоит непосредственно перед `before` в исходном тексте — модель,
# вероятно, исправляет реальную ошибку управления, и пункт нужно
# сохранить (например, «согласно приказа» → «согласно приказу»).
#
# Список консервативный: туда включены только те предлоги, которые
# **строго** управляют падежом и часто становятся источником ошибок
# управления в деловом тексте. Двусмысленные предлоги («в», «на», «по»
# имеют несколько падежных «лиц») мы тоже включаем — лучше пропустить
# реальную правку, чем сломать её.
_CASE_GOVERNING_PREPS = frozenset({
    # дательный падеж
    "согласно", "благодаря", "вопреки", "наперекор", "подобно",
    "соразмерно", "соответственно", "к", "ко",
    # винительный падеж
    "через", "сквозь", "про", "несмотря",
    # родительный падеж
    "из", "от", "до", "без", "у", "вблизи", "вместо", "возле",
    "вокруг", "впереди", "ввиду", "вне", "внутри",
    "касательно", "касаемо", "кроме", "мимо", "напротив", "около",
    "относительно", "поверх", "подле", "позади", "после", "посреди",
    "посредством", "против", "путём", "путем", "ради", "сверх", "среди",
    # творительный падеж
    "над", "перед", "пред",
    # предложный падеж
    "при", "о", "об", "обо",
    # многозначные (включаем для безопасности)
    "в", "во", "на", "по", "с", "со", "под", "между", "за",
})


class MorphFilter:
    """Фильтр падежных «улучшений» через pymorphy3.

    Конструктор лениво пытается импортировать `pymorphy3`. Если не
    получилось — `available` остаётся False, и все методы возвращают
    «не фильтровать» (no-op). Это позволяет PR откатить через простое
    удаление зависимости из requirements без правки кода.
    """

    def __init__(self):
        self._morph = None
        try:
            import pymorphy3  # type: ignore[import-untyped]
            self._morph = pymorphy3.MorphAnalyzer()
            _log.info("MorphFilter: pymorphy3 загружен (готов фильтровать падежные улучшения)")
        except Exception as e:
            _log.warning(
                "MorphFilter: pymorphy3 не загружен (%s) — "
                "фильтр падежных «улучшений» отключён",
                e,
            )

    @property
    def available(self) -> bool:
        return self._morph is not None

    # POS, требующие согласования с существительным (case+number+gender).
    # Если перед `before` стоит слово такой POS, и НИ ОДИН парс `before`
    # не согласуется с НИ ОДНИМ парсом этого слова — `before` грамматически
    # рассогласован с контекстом, т.е. реальная ошибка, а не валидная форма.
    _POS_REQUIRING_AGREEMENT = frozenset({
        "ADJF",   # полное прилагательное
        "ADJS",   # краткое прилагательное
        "PRTF",   # полное причастие
        "PRTS",   # краткое причастие
        "NPRO",   # местоимение-существительное
    })

    def _is_preposition_word(self, word: str) -> bool:
        """v1.8.5: True если у слова есть PREP-парс. Используется чтобы
        обнаружить, что предыдущее слово (prev_word) на самом деле
        управляется предшествующим предлогом, а не модифицирует
        следующее существительное. См. `_is_grammatically_disagreed_with_prev`.
        """
        if self._morph is None or not word:
            return False
        try:
            parses = self._morph.parse(word.lower())
        except Exception:
            return False
        return any("PREP" in str(p.tag) for p in parses)

    def _is_grammatically_disagreed_with_prev(
        self, before: str, raw_text: str,
        parsed_doc: Optional["ParsedDoc"] = None,
    ) -> bool:
        """v1.7.3: True если в `raw_text` непосредственно перед `before`
        стоит ADJF/ADJS/PRTF/PRTS/NPRO, и **ни один** парс `before` не
        согласуется с **ни одним** парсом этого предыдущего слова по
        case+number+gender. Это означает что `before` — реальная
        грамматическая ошибка согласования, а не валидная падежная форма.

        В таких случаях правка модели (`before` → `after`) — legitimate
        number/gender/case fix, а не галлюцинация, и мы НЕ должны
        её откатывать.

        Главный кейс: «Проверочное мероприятия» (ADJF.neut.sing →
        NOUN.neut.{gent.sing | plur.nomn | plur.accs}) — ни один парс
        «мероприятия» не согласован с «Проверочное» (для каждого парса
        либо разное число, либо разный падеж). Значит «Проверочное
        мероприятия» — disagreement, и правка на «мероприятие»
        (sing.nomn.neut) — legitimate.

        Анти-кейс: «ущерба Подразделения» — «ущерба» это NOUN (не в
        POS_REQUIRING_AGREEMENT), поэтому функция возвращает False
        (нет проверки), и стандартная case-only-логика идёт в работу.
        """
        if not raw_text or self._morph is None:
            return False
        idx = raw_text.find(before)
        if idx < 0:
            return False
        # v1.9: если есть дерево зависимостей — спрашиваем у него
        # синтаксическую связь между prev_word и before на этом offset.
        # Если НЕ attributive (amod/det/...) или prev сам управляется
        # предлогом — disagreement НЕ существует (дерево говорит
        # что эти слова не должны согласовываться). Возвращаем False,
        # это включит стандартный case-only-фильтр и заблокирует
        # hallucinated case-change от T-lite. Сигнал True (`рассогласованы`)
        # возвращаем только при attributive-связи с реальным
        # морфологическим несовпадением.
        if parsed_doc is not None:
            noun_idx = parsed_doc.token_at_offset(idx)
            if noun_idx is not None:
                # Ищем предыдущий по дереву токен (не PUNCT) и смотрим
                # его связь с noun_idx.
                prev_idx = noun_idx - 1
                while prev_idx >= 0 and parsed_doc.tokens[prev_idx].pos == "PUNCT":
                    prev_idx -= 1
                if prev_idx >= 0:
                    if parsed_doc.is_clearly_non_attributive(prev_idx, noun_idx):
                        # Синтаксически эти слова ЯВНО НЕ в attributive-
                        # связи (либо obl:agent, либо case-governed
                        # через case-child, либо parataxis). disagreement
                        # не существует. Возвращаем False — пусть
                        # стандартный case-only-фильтр заблокирует hallucination.
                        return False
                    # Иначе (включая случаи где natasha мис-парсит):
                    # fallback на pymorphy3-морфоcheck ниже + v1.8.5
                    # hardcoded skip-rules.
        prefix = raw_text[:idx].rstrip()
        m = re.search(r"(\S+)\s*$", prefix)
        if not m:
            return False
        prev_word = m.group(1).strip(".,;:!?\"'«»()[]{}—–-")
        if not prev_word:
            return False
        # v1.8.5: смотрим ещё на одно слово назад — если prev_word сам
        # управляется предлогом («при этом», «в том», «о тех»...), то
        # prev_word — объект предлога, а не модификатор `before`. Иначе
        # disagreement-логика даст ложноположительный «before рассогласован»
        # на discourse markers.
        prev_prefix = prefix[: prefix.rfind(prev_word)].rstrip() if prev_word in prefix else ""
        prev_prev_match = re.search(r"(\S+)\s*$", prev_prefix) if prev_prefix else None
        prev_prev_word = (
            prev_prev_match.group(1).strip(".,;:!?\"'«»()[]{}—–-")
            if prev_prev_match else ""
        )
        try:
            prev_parses = self._morph.parse(prev_word)
            b_parses = self._morph.parse(before)
        except Exception:
            return False
        if not prev_parses or not b_parses:
            return False
        # Только парсы предыдущего слова, требующие agreement с noun
        prev_agree = [p for p in prev_parses
                      if p.tag.POS in self._POS_REQUIRING_AGREEMENT]
        if not prev_agree:
            return False
        # v1.8.5: если prev — причастие (PRTF/PRTS), а before может быть
        # в творительном (ablt) — это паттерн «причастие + агенс_в_творительном»
        # («проводимых подразделениями», «выполненная Ивановым»), валидная
        # русская конструкция. Причастие согласуется не с этим существительным,
        # а с upstream-головой («преступлений, проводимых X»), и НЕТ
        # рассогласования. Возвращаем False — это НЕ disagreement, и
        # `is_case_only_substitution` дальше отработает стандартный case-only
        # тест → заблокирует hallucinated fix «подразделениями→подразделений».
        prev_is_participle = any(
            p.tag.POS in ("PRTF", "PRTS") for p in prev_parses
        )
        before_can_be_ablt = any(p.tag.case == "ablt" for p in b_parses)
        if prev_is_participle and before_can_be_ablt:
            return False
        # v1.8.5: если перед prev_word стоит предлог (PREP) — prev_word
        # это объект предлога («при этом», «о том», «в этих»...), а НЕ
        # модификатор before. Прод-кейс 05.05.2026: «при этом количество
        # должностных преступлений — 47» → T-lite сочиняет fix
        # «количество → количестве», и фильтр пропускает его, так как
        # видит «этом»(loct) перед «количество»(nomn) и считает это
        # disagreement. На самом деле «этом» бнут к «при», и «количество»
        # начинает новую clause — disagreement-нет.
        if prev_prev_word and self._is_preposition_word(prev_prev_word):
            return False
        # Хотя бы одна пара (prev_p, b_p) согласована? Тогда before — валиден.
        for prev_p in prev_agree:
            for b_p in b_parses:
                if (prev_p.tag.case is not None
                        and prev_p.tag.case == b_p.tag.case
                        and prev_p.tag.number == b_p.tag.number
                        and prev_p.tag.gender == b_p.tag.gender):
                    return False  # есть согласование → before валиден
        # Ни один парс before не согласуется с prev → disagreement
        return True

    def is_case_only_substitution(
        self, before: str, after: str, raw_text: Optional[str] = None,
        parsed_doc: Optional["ParsedDoc"] = None,
    ) -> bool:
        """True если `before` и `after` — формы одной леммы, отличающиеся
        ТОЛЬКО падежом (не числом, не родом, не POS).

        Реализация (v1.7.3 — устраняет false-positive на амбигуитете
        pymorphy3):
          1. Оба должны быть однословными непустыми токенами.
          2. После нормализации ё/е они НЕ должны совпадать (тогда это
             ё-фильтр, а не наш).
          3. Если передан `raw_text` и `before` грамматически
             рассогласован с предыдущим adj/прич/мест в этом тексте
             (т.е. ни один парс `before` не согласован с предыдущим
             словом) — это реальная ошибка согласования, не case-only.
             Возвращаем False (не блокируем legitimate fix).
          4. Best-парсы pymorphy3 у обоих должны иметь одинаковую
             лемму и одинаковую POS.
          5. У обоих должен быть `case`.
          6. `number` best-парсов должен совпадать.
          7. `case` best-парсов должен различаться.

        Главное отличие от v1.7/v1.7.1: при наличии `raw_text` сначала
        проверяется adj-noun агремент с предыдущим словом. Это
        позволяет различить:
          * «Проверочное (sing) мероприятия (plur)» — disagreement,
            модель fixит на «мероприятие» (sing) — legitimate, НЕ блокируем;
          * «ущерба Подразделения» — управление noun-noun, без
            agreement-проверки идёт стандартная case-only логика и
            «Подразделения → Подразделению» блокируется как раньше.
        """
        if self._morph is None:
            return False
        b = before.strip()
        a = after.strip()
        if not b or not a:
            return False
        if " " in b or " " in a:
            return False
        if b.lower().replace("ё", "е") == a.lower().replace("ё", "е"):
            return False
        # Контекстная проверка: если before рассогласован с предыдущим
        # adj/прич/мест → реальная ошибка, не блокируем.
        if raw_text and self._is_grammatically_disagreed_with_prev(
            b, raw_text, parsed_doc=parsed_doc,
        ):
            return False
        try:
            pb = self._morph.parse(b)
            pa = self._morph.parse(a)
        except Exception:
            return False
        if not pb or not pa:
            return False
        bp = pb[0]
        ap = pa[0]
        if bp.normal_form != ap.normal_form:
            return False
        if not bp.tag.case or not ap.tag.case:
            return False
        if bp.tag.number != ap.tag.number:
            return False
        if bp.tag.case == ap.tag.case:
            return False
        if bp.tag.POS != ap.tag.POS:
            return False
        return True

    @staticmethod
    def has_case_governing_context(before: str, raw_text: str) -> bool:
        """True если непосредственно перед `before` в `raw_text` стоит
        предлог/наречие, требующее конкретного падежа.

        Использует первую находку `before` в `raw_text`. Это
        достаточно для типовых документов — модель редко правит одно и
        то же слово в разных контекстах с разными управляющими словами.
        """
        if not before or not raw_text:
            return False
        idx = raw_text.find(before)
        if idx < 0:
            return False
        prefix = raw_text[:idx].rstrip()
        m = re.search(r"(\S+)\s*$", prefix)
        if not m:
            return False
        prev = m.group(1).lower().strip(".,;:!?\"'«»()[]{}")
        return prev in _CASE_GOVERNING_PREPS

    def is_hallucinated_case_change(
        self, before: str, after: str, raw_text: str,
        parsed_doc: Optional["ParsedDoc"] = None,
    ) -> bool:
        """Главный публичный метод. Возвращает True если правка
        `before` → `after` — галлюцинированное «улучшение» уже валидной
        падежной формы (одна лемма + тот же number + разный case + НЕТ
        управляющего предлога перед `before` в `raw_text`).

        v1.7.3: передаёт `raw_text` в `is_case_only_substitution`, чтобы
        тот мог проверить adj-noun агремент и не блокировать legitimate
        number-фиксы (см. `_is_grammatically_disagreed_with_prev`).

        Используется в серверном пайплайне для дропа таких пунктов из
        ===CHANGES=== и отката подмены в ===CORRECTED===.
        """
        # v1.9: если не передан parsed_doc и NATASHA_ENABLED=true — пытаемся
        # получить дерево лениво (кэшируется, так что один раз на
        # raw_text). Пробрасываем в is_case_only_substitution →
        # _is_grammatically_disagreed_with_prev.
        if parsed_doc is None and _NATASHA_ENABLED_ENV:
            try:
                from shared.syntax_parser import get_syntax_parser  # noqa: E402
                parser = get_syntax_parser()
                if parser.available:
                    parsed_doc = parser.parse(raw_text)
            except Exception as exc:  # pragma: no cover
                _log.warning(
                    "MorphFilter: не удалось загрузить syntax_parser: %s",
                    exc,
                )
        if not self.is_case_only_substitution(before, after, raw_text, parsed_doc):
            return False
        if self.has_case_governing_context(before, raw_text):
            return False
        return True

    def find_hallucinated_pairs_in_compound(
        self, before: str, after: str, raw_text: str
    ) -> list[tuple[str, str]]:
        """v1.7.1: для compound-кейсов вида «before phrase» → «after phrase»
        (модель упаковывает несколько правок в одну цитату) — возвращает
        список (b_word, a_word) пар, в которых b_word/a_word отличаются
        и являются галлюцинированной падежной подменой.

        Главный prod-кейс (КС-2, 6 мая 2026):
          before = «повлекших риски причинения ущерба Подразделения»
          after  = «повлёкших риски причинения ущерба Подразделению»
        Возвращает [("Подразделения", "Подразделению")] —
        («повлекших», «повлёкших») это ё-разница (handles by eyo-undo
        elsewhere), не case substitution; остальные слова идентичны.

        Если before и after — одиночные слова (нет пробелов), вернёт
        либо [(before, after)] если is_hallucinated_case_change(before,
        after, raw_text) — это совместимо с старым single-word методом
        — либо пустой список.

        Если число токенов в before и after различается — вернёт пустой
        список (compound с inserted/deleted словами, не case-only
        substitution).
        """
        if self._morph is None:
            return []
        b = (before or "").strip()
        a = (after or "").strip()
        if not b or not a:
            return []
        b_tokens = b.split()
        a_tokens = a.split()
        if len(b_tokens) != len(a_tokens):
            return []
        out: list[tuple[str, str]] = []
        for bw, aw in zip(b_tokens, a_tokens):
            if bw == aw:
                continue
            # Уже обработано ё-фильтром в CORRECTED: пропускаем (но
            # не считаем «реальной» разницей).
            if bw.lower().replace("ё", "е") == aw.lower().replace("ё", "е"):
                continue
            if self.is_hallucinated_case_change(bw, aw, raw_text):
                out.append((bw, aw))
        return out

    def is_compound_fully_hallucinated(
        self, before: str, after: str, raw_text: str
    ) -> bool:
        """True если ВСЕ нетривиальные различия между before и after
        являются галлюцинированными падежными подменами (или ё-только
        различиями, обрабатываемыми отдельно ё-фильтром).

        Используется в `_drop_morph_case_substitutions` для решения:
        дропать ли весь пункт ===CHANGES=== (все различия фантомные)
        или только откатить подмены отдельных слов в ===CORRECTED===
        (есть и реальные правки, оставляем пункт).
        """
        if self._morph is None:
            return False
        b_tokens = (before or "").strip().split()
        a_tokens = (after or "").strip().split()
        if not b_tokens or not a_tokens:
            return False
        if len(b_tokens) != len(a_tokens):
            return False
        has_diff = False
        for bw, aw in zip(b_tokens, a_tokens):
            if bw == aw:
                continue
            has_diff = True
            # ё-only различие — считается «обрабатывается отдельно»,
            # не блокирует дроп
            if bw.lower().replace("ё", "е") == aw.lower().replace("ё", "е"):
                continue
            if not self.is_hallucinated_case_change(bw, aw, raw_text):
                return False
        return has_diff


# Singleton для переиспользования между запросами (pymorphy3 загружает
# словари ~50 МБ — повторно делать дорого). Создаётся лениво первым
# вызовом get_morph_filter().
_singleton: Optional[MorphFilter] = None


def get_morph_filter() -> MorphFilter:
    """Возвращает singleton MorphFilter. Lazily инициализируется."""
    global _singleton
    if _singleton is None:
        _singleton = MorphFilter()
    return _singleton
