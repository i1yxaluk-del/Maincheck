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
import re
from typing import Optional

_log = logging.getLogger("ai_suggester.morph_filter")

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

    def is_case_only_substitution(self, before: str, after: str) -> bool:
        """True если `before` и `after` — формы одной леммы, отличающиеся
        только падежом (не числом, не родом, не POS).

        Реализация:
          1. Оба должны быть однословными непустыми токенами.
          2. После нормализации ё/е они НЕ должны совпадать (тогда это
             ё-фильтр, а не наш).
          3. Лучшие парсы pymorphy3 у обоих должны иметь одинаковую
             лемму и одинаковую POS.
          4. У обоих должен быть `case` (склоняемые слова).
          5. `number` должен совпадать (sing↔sing или plur↔plur).
          6. `case` должен различаться.
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

    def is_hallucinated_case_change(self, before: str, after: str, raw_text: str) -> bool:
        """Главный публичный метод. Возвращает True если правка
        `before` → `after` — галлюцинированное «улучшение» уже валидной
        падежной формы (одна лемма + тот же number + разный case + НЕТ
        управляющего предлога перед `before` в `raw_text`).

        Используется в серверном пайплайне для дропа таких пунктов из
        ===CHANGES=== и отката подмены в ===CORRECTED===.
        """
        if not self.is_case_only_substitution(before, after):
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
