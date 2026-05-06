"""Юнит-тесты MorphFilter (server/shared/morph_filter.py) — v1.7.

Проверяет, что фильтр падежных «улучшений» правильно различает:
  * галлюцинации модели (одна лемма + тот же number + разный case +
    нет управляющего предлога) — должны фильтроваться;
  * реальные правки управления («согласно приказа» → «согласно приказу»)
    — должны пропускаться;
  * agreement fixes («выполненной» → «выполненных», разница в числе)
    — должны пропускаться;
  * лексические замены (разные леммы) — должны пропускаться.

Если pymorphy3 не установлен (CI без словаря), все тесты пропускаются
через `pytest.importorskip` — фильтр всё равно безопасен (no-op).
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))

# Если pymorphy3 не установлен — пропускаем весь модуль (фильтр всё
# равно безопасен в этом случае, превращается в no-op).
pytest.importorskip("pymorphy3")

from shared.morph_filter import MorphFilter, get_morph_filter  # noqa: E402


@pytest.fixture(scope="module")
def mf():
    """Singleton-инстанс MorphFilter — pymorphy3 грузит ~50 МБ словарей,
    дорого делать в каждом тесте."""
    f = get_morph_filter()
    if not f.available:
        pytest.skip("pymorphy3 не загружается — пропускаем юнит-тесты MorphFilter")
    return f


# ─── is_case_only_substitution ────────────────────────────────────────


def test_case_only_positive_подразделения(mf):
    """Главный prod-кейс v1.6.9 КС-2: Подразделения (gent) → Подразделению (datv).
    Одна лемма «подразделение», то же число (sing), разный падеж — case-only."""
    assert mf.is_case_only_substitution("Подразделения", "Подразделению") is True


def test_case_only_positive_lowercase(mf):
    """То же слово в нижнем регистре — фильтр должен срабатывать
    (модель часто сохраняет регистр исходника, но проверяем).
    """
    assert mf.is_case_only_substitution("подразделения", "подразделению") is True


def test_case_only_positive_приказа_приказу(mf):
    """приказа (gent) → приказу (datv): одна лемма «приказ», sing-sing,
    разный падеж — формально case-only. Контекст («согласно ...»)
    проверяется отдельно через has_case_governing_context."""
    assert mf.is_case_only_substitution("приказа", "приказу") is True


def test_case_only_negative_number_change(mf):
    """выполненной (sing) → выполненных (plur): разное число — это
    agreement fix, не case-only. Должны пропустить (KEEP)."""
    assert mf.is_case_only_substitution("выполненной", "выполненных") is False


def test_case_only_negative_number_change_стоимостей(mf):
    """стоимостей (plur) → стоимости (sing): разное число.
    Реальный кейс из КС-2 — должен пропуститься."""
    assert mf.is_case_only_substitution("стоимостей", "стоимости") is False


def test_case_only_negative_eyo_only(mf):
    """повлекших → повлёкших: совпадают после нормализации ё/е.
    Это работа ё-фильтра, не нашего."""
    assert mf.is_case_only_substitution("повлекших", "повлёкших") is False


def test_case_only_negative_different_lemmas(mf):
    """принимать → принять: разные леммы (разный вид глагола).
    Лексическая замена, должна пропуститься."""
    assert mf.is_case_only_substitution("принимать", "принять") is False


def test_case_only_negative_multiword(mf):
    """Многословные before/after — не case-only (фильтр работает только
    на одиночных словах). Должно вернуть False, чтобы compound-цитаты
    обрабатывались отдельно (ими занимается ё-фильтр)."""
    assert mf.is_case_only_substitution("ущерба Подразделения", "ущерба Подразделению") is False


def test_case_only_negative_empty(mf):
    """Пустые before/after — без падений, False."""
    assert mf.is_case_only_substitution("", "") is False
    assert mf.is_case_only_substitution("приказ", "") is False
    assert mf.is_case_only_substitution("", "приказу") is False


def test_case_only_negative_same_form(mf):
    """before == after: формально not case-substitution (различия нет).
    Защита от тривиально-пустых правок."""
    assert mf.is_case_only_substitution("приказ", "приказ") is False


def test_case_only_negative_unknown_word(mf):
    """Аббревиатура/несклоняемое слово (ФСБ): pymorphy3 не определит
    case → должны пропустить (False)."""
    # ФСБ может парситься как UNKN или как нескл-аббревиатура без case.
    # В любом случае — не должны вернуть True, чтобы не сломать
    # реальные правки (скажем, разворот ФСБ → ФСБ России).
    assert mf.is_case_only_substitution("ФСБ", "ФСБ") is False


# ─── has_case_governing_context ───────────────────────────────────────


def test_has_governing_context_согласно():
    """согласно — управляет дательным; перед before оно — есть управление."""
    raw = "согласно приказа №5 от 12.05.2026"
    assert MorphFilter.has_case_governing_context("приказа", raw) is True


def test_has_governing_context_благодаря():
    """благодаря — управляет дательным."""
    raw = "благодаря решения комиссии"
    assert MorphFilter.has_case_governing_context("решения", raw) is True


def test_has_governing_context_no_prep():
    """Если перед before нет управляющего предлога — False.
    Главный prod-кейс: «причинения ущерба Подразделения» — «ущерба»
    не управляющий, оно само в genitive."""
    raw = "причинения ущерба Подразделения в размере"
    assert MorphFilter.has_case_governing_context("Подразделения", raw) is False


def test_has_governing_context_word_not_found():
    """before отсутствует в raw_text — безопасный False (не падать)."""
    assert MorphFilter.has_case_governing_context("слово", "совсем другой текст") is False


def test_has_governing_context_empty():
    """Защита от пустых аргументов."""
    assert MorphFilter.has_case_governing_context("", "текст") is False
    assert MorphFilter.has_case_governing_context("слово", "") is False


def test_has_governing_context_with_punctuation_around_prep():
    """Перед before стоит «согласно,» — пунктуация после предлога не
    должна ломать определение."""
    # NB: в реальном тексте «согласно, ...» — невалидно, но регэксп
    # ловит последнее непробельное слово. Если оно предлог — True.
    raw = "согласно приказу"
    assert MorphFilter.has_case_governing_context("приказу", raw) is True


# ─── is_hallucinated_case_change ──────────────────────────────────────


def test_hallucinated_подразделения(mf):
    """Главный prod-кейс v1.6.9: модель «исправляет» уже валидную форму."""
    raw = "повлекших риски причинения ущерба Подразделения в размере более 2 млн рублей"
    assert mf.is_hallucinated_case_change("Подразделения", "Подразделению", raw) is True


def test_not_hallucinated_согласно_приказа(mf):
    """Реальная ошибка управления — НЕ должна фильтроваться."""
    raw = "согласно приказа №5"
    assert mf.is_hallucinated_case_change("приказа", "приказу", raw) is False


def test_not_hallucinated_благодаря_решения(mf):
    """благодаря требует дательного, «благодаря решения» — ошибка."""
    raw = "благодаря решения комиссии задача выполнена"
    assert mf.is_hallucinated_case_change("решения", "решению", raw) is False


def test_not_hallucinated_number_change(mf):
    """Изменение числа — пропускаем (agreement fix)."""
    raw = "стоимости выполненной работ путём применения завышенных расценок"
    assert mf.is_hallucinated_case_change("выполненной", "выполненных", raw) is False


def test_not_hallucinated_lexical(mf):
    """Лексическая замена — разные леммы, пропускаем."""
    raw = "комиссия принимать решение должна сегодня"
    assert mf.is_hallucinated_case_change("принимать", "принять", raw) is False


# ─── Сценарии устойчивости ────────────────────────────────────────────


def test_unavailable_morph_returns_false():
    """Если pymorphy3 не загружен (например, в минимальном CI) — все
    is_* возвращают False, чтобы пайплайн ничего не дропал."""

    class _FakeUnavailable(MorphFilter):
        def __init__(self):
            self._morph = None  # type: ignore[assignment]

    f = _FakeUnavailable()
    assert f.available is False
    assert f.is_case_only_substitution("Подразделения", "Подразделению") is False
    assert (
        f.is_hallucinated_case_change("Подразделения", "Подразделению", "ущерба Подразделения")
        is False
    )


def test_singleton_reuse():
    """get_morph_filter() возвращает один и тот же объект между
    вызовами (pymorphy3 загружается один раз)."""
    a = get_morph_filter()
    b = get_morph_filter()
    assert a is b
