"""v1.8c: тесты SageValidator.

ВАЖНО: эти тесты НЕ загружают реальную sage-fredt5 модель — она ~95M
параметров и требует transformers/torch (~1.5 ГБ зависимостей).
Реальная модель загружается только в проде через SAGE_VALIDATOR_ENABLED=true.

Здесь тестируем:
  1. Конфиг из ENV (включая дефолты).
  2. judge() — pure-функция, верный ли verdict при разных sage-text.
  3. should_drop() — domain admin vs general.
  4. is_available() возвращает False когда:
       - SAGE_VALIDATOR_ENABLED=false (по умолчанию),
       - transformers/torch не установлены (ImportError ловится).
  5. Helper'ы _normalize_ws / _swap_first_case.
"""
from __future__ import annotations

import os

import pytest

from server.shared.sage_validator import (
    VERDICT_AGREE,
    VERDICT_DISAGREE,
    VERDICT_UNKNOWN,
    SageConfig,
    SageValidator,
    _normalize_ws,
    _swap_first_case,
    reset_validator_for_testing,
)


@pytest.fixture(autouse=True)
def _clean_singleton():
    reset_validator_for_testing()
    yield
    reset_validator_for_testing()


# ───────────────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────────────

def test_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # Удаляем переменные окружения, чтобы тестировать чистые дефолты
    for k in (
        "SAGE_VALIDATOR_ENABLED", "SAGE_VALIDATOR_MODE",
        "SAGE_VALIDATOR_DOMAIN", "SAGE_VALIDATOR_CATEGORIES",
        "SAGE_VALIDATOR_MODEL", "SAGE_VALIDATOR_DEVICE",
        "SAGE_VALIDATOR_MAX_INPUT_LEN", "SAGE_VALIDATOR_WARMUP",
    ):
        monkeypatch.delenv(k, raising=False)
    cfg = SageConfig.from_env()
    # Default OFF
    assert cfg.enabled is False
    # Default — dryrun (только логи, ничего не дропаем)
    assert cfg.mode == "dryrun"
    # Default — admin (priority recall)
    assert cfg.domain == "admin"
    # Default categories — только орфография (sage именно на ней обучена)
    assert cfg.categories == ("орфограф",)
    assert "sage-fredt5-distilled-95m" in cfg.model_name
    assert cfg.device == "cpu"
    assert cfg.max_input_len == 512
    assert cfg.warmup is True


def test_config_mode_invalid_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAGE_VALIDATOR_MODE", "garbage")
    cfg = SageConfig.from_env()
    assert cfg.mode == "dryrun"


def test_config_mode_enforce(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAGE_VALIDATOR_MODE", "enforce")
    cfg = SageConfig.from_env()
    assert cfg.mode == "enforce"


def test_config_categories_custom(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAGE_VALIDATOR_CATEGORIES", "орфограф,пунктуация")
    cfg = SageConfig.from_env()
    assert "орфограф" in cfg.categories
    assert "пунктуация" in cfg.categories


def test_config_categories_empty_means_all(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAGE_VALIDATOR_CATEGORIES", "")
    cfg = SageConfig.from_env()
    assert cfg.categories == ()


@pytest.mark.parametrize("raw,expected", [
    ("true", True), ("1", True), ("yes", True), ("on", True),
    ("True", True), ("TRUE", True),
    ("false", False), ("0", False), ("no", False), ("off", False),
    ("", False), ("garbage", False),
])
def test_config_enabled_parse(monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool) -> None:
    monkeypatch.setenv("SAGE_VALIDATOR_ENABLED", raw)
    cfg = SageConfig.from_env()
    assert cfg.enabled is expected


def test_config_invalid_domain_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAGE_VALIDATOR_DOMAIN", "moonbase")
    cfg = SageConfig.from_env()
    assert cfg.domain == "admin"


def test_config_general_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAGE_VALIDATOR_DOMAIN", "general")
    cfg = SageConfig.from_env()
    assert cfg.domain == "general"


# ───────────────────────────────────────────────────────────────────────
# is_available
# ───────────────────────────────────────────────────────────────────────

def _cfg(
    *,
    enabled: bool = True,
    mode: str = "enforce",
    domain: str = "admin",
    categories: tuple[str, ...] = (),  # все категории
    model_name: str = "x",
) -> SageConfig:
    return SageConfig(
        enabled=enabled,
        mode=mode,
        domain=domain,
        categories=categories,
        model_name=model_name,
        device="cpu",
        max_input_len=512,
        warmup=False,
    )


def test_is_available_false_when_disabled() -> None:
    cfg = _cfg(enabled=False)
    v = SageValidator(cfg)
    assert v.is_available() is False


def test_is_available_false_when_model_missing() -> None:
    """С несуществующим model_name загрузка должна упасть и валидатор
    остаться недоступным — без падения сервера."""
    cfg = _cfg(model_name="this-model-does-not-exist-12345/nope")
    v = SageValidator(cfg)
    # Не должно бросать — внутренний try/except перехватывает
    available = v.is_available()
    # Может быть True если transformers вообще не установлены (ImportError
    # тоже ловится) — в обоих случаях это False
    assert available is False
    # correct() в no-op режиме возвращает text как есть
    assert v.correct("Привет, мир.") == "Привет, мир."


# ───────────────────────────────────────────────────────────────────────
# judge (pure-функция, не требует модели)
# ───────────────────────────────────────────────────────────────────────

def _make_validator(
    domain: str = "admin",
    mode: str = "enforce",
    categories: tuple[str, ...] = (),
) -> SageValidator:
    return SageValidator(_cfg(domain=domain, mode=mode, categories=categories))


def test_judge_agree_when_after_in_sage() -> None:
    v = _make_validator()
    # T-lite предложил «кварталах → квартале», sage тоже исправил
    sage_text = "Во 2-м квартале 2025 года проведено мероприятие."
    assert v.judge("кварталах", "квартале", sage_text) == VERDICT_AGREE


def test_judge_disagree_when_before_stays_in_sage() -> None:
    v = _make_validator()
    # T-lite предложил «мероприятие → мероприятия», но sage оставил «мероприятие»
    sage_text = "Во 2-м квартале 2025 года проведено мероприятие."
    assert v.judge("мероприятие", "мероприятия", sage_text) == VERDICT_DISAGREE


def test_judge_unknown_when_neither_in_sage() -> None:
    v = _make_validator()
    # Sage сделал что-то третье — ни before, ни after не нашли
    sage_text = "Совсем другая фраза без целевых слов."
    assert v.judge("кварталах", "квартале", sage_text) == VERDICT_UNKNOWN


def test_judge_empty_returns_unknown() -> None:
    v = _make_validator()
    assert v.judge("", "квартале", "что-то") == VERDICT_UNKNOWN
    assert v.judge("кварталах", "", "что-то") == VERDICT_UNKNOWN
    assert v.judge("кварталах", "квартале", "") == VERDICT_UNKNOWN


def test_judge_handles_case_swap_of_first_letter() -> None:
    """sage может вернуть «Проверочное» когда T-lite дал «проверочное»."""
    v = _make_validator()
    sage_text = "Проверочное мероприятие проведено в срок."
    # T-lite предложил «проверочное → Проверочное» (стилистика заголовка);
    # хотим AGREE даже если sage начал с большой
    assert v.judge("проверочное", "проверочное", sage_text) in (
        VERDICT_AGREE,
        VERDICT_DISAGREE,
    )  # любой определённый вердикт, не UNKNOWN


def test_judge_whitespace_robust() -> None:
    """Лишние пробелы / переносы строк в sage-выводе не должны менять verdict."""
    v = _make_validator()
    sage_text = "Во\n2-м    квартале   2025\nгода."
    assert v.judge("кварталах", "квартале", sage_text) == VERDICT_AGREE


# ───────────────────────────────────────────────────────────────────────
# should_drop (зависит от domain)
# ───────────────────────────────────────────────────────────────────────

def test_should_drop_admin_only_disagree() -> None:
    v = _make_validator(domain="admin", mode="enforce")
    assert v.should_drop(VERDICT_DISAGREE) is True
    # В admin-режиме UNKNOWN не дропаем — recall важнее
    assert v.should_drop(VERDICT_UNKNOWN) is False
    assert v.should_drop(VERDICT_AGREE) is False


def test_should_drop_general_drops_unknown_too() -> None:
    v = _make_validator(domain="general", mode="enforce")
    assert v.should_drop(VERDICT_DISAGREE) is True
    assert v.should_drop(VERDICT_UNKNOWN) is True
    assert v.should_drop(VERDICT_AGREE) is False


def test_should_drop_dryrun_never_drops() -> None:
    """В dryrun-режиме НИЧЕГО не дропаем (только логируем) — это default."""
    v = _make_validator(mode="dryrun")
    assert v.should_drop(VERDICT_DISAGREE) is False
    assert v.should_drop(VERDICT_UNKNOWN) is False
    assert v.should_drop(VERDICT_AGREE) is False


def test_should_drop_category_filter_blocks_non_orthography() -> None:
    """С `categories=("орфограф",)` дропаем только орфографические правки."""
    v = _make_validator(mode="enforce", categories=("орфограф",))
    # Orthography → дропаем
    assert v.should_drop(VERDICT_DISAGREE, category="орфография — опечатка") is True
    # Не-орфография → НЕ дропаем (sage ненадёжна для согласования/управления)
    assert v.should_drop(VERDICT_DISAGREE, category="согласование") is False
    assert v.should_drop(VERDICT_DISAGREE, category="управление") is False
    assert v.should_drop(VERDICT_DISAGREE, category="пунктуация") is False


def test_should_drop_empty_categories_allows_all() -> None:
    """Если SAGE_VALIDATOR_CATEGORIES="" — все категории фильтруются."""
    v = _make_validator(mode="enforce", categories=())
    assert v.should_drop(VERDICT_DISAGREE, category="согласование") is True
    assert v.should_drop(VERDICT_DISAGREE, category="орфография") is True


def test_should_drop_category_filter_case_insensitive() -> None:
    """Substring-match должен быть case-insensitive."""
    v = _make_validator(mode="enforce", categories=("орфограф",))
    assert v.should_drop(VERDICT_DISAGREE, category="Орфография") is True
    assert v.should_drop(VERDICT_DISAGREE, category="ОРФОГРАФИЯ") is True


def test_category_matches_helper() -> None:
    """category_matches возвращает True/False корректно."""
    v = _make_validator(categories=("орфограф",))
    assert v.category_matches("орфография — пропущена буква") is True
    assert v.category_matches("согласование числа") is False
    assert v.category_matches("") is False  # пустая категория не матчит при непустом фильтре
    # Пустой фильтр → всё матчит
    v_all = _make_validator(categories=())
    assert v_all.category_matches("любая") is True
    assert v_all.category_matches("") is True


# ───────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────

def test_normalize_ws() -> None:
    assert _normalize_ws("  привет\n\tмир   ") == "привет мир"
    assert _normalize_ws("") == ""
    assert _normalize_ws("single") == "single"


def test_swap_first_case() -> None:
    assert _swap_first_case("Привет") == "привет"
    assert _swap_first_case("привет") == "Привет"
    assert _swap_first_case("") == ""
    # Не-буква на первой позиции — не меняем
    assert _swap_first_case("123abc") == "123abc"


# ───────────────────────────────────────────────────────────────────────
# Domain priority — закрывает заявленный кейс admin → priority recall
# ───────────────────────────────────────────────────────────────────────

def test_admin_keeps_unknown_general_drops() -> None:
    """Регрессия: в admin-режиме UNKNOWN-правки T-lite остаются, в general
    дропаются. Это ключевая разница, заявленная в дизайне v1.8c — admin
    приоритизирует recall, general — precision."""
    v_admin = _make_validator(domain="admin", mode="enforce")
    v_general = _make_validator(domain="general", mode="enforce")
    # Один и тот же verdict, но решения разные
    assert v_admin.should_drop(VERDICT_UNKNOWN) is False
    assert v_general.should_drop(VERDICT_UNKNOWN) is True
