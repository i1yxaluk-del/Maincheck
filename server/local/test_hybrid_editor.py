"""Инварианты безопасного diff-а (сохранены из v2.4, обновлены под v9)."""

from decision_engine import DecisionEngine
from hybrid_editor import STACKS
from safe_diff import diff_candidates
from verification import GenerativeGuard

REGRESSION = (
    "План профессиональной служебная и физической подготовки на 2026 учебных "
    "год требует, корректировки (штабная тренировка спланирована в один день "
    "с тактико-специальным занятием с элементами командно-штабной тренировки "
    "по одна и той же теме);"
)


def test_preset_contract():
    assert set(STACKS) == {"A", "B", "X", "Y"}
    assert STACKS["A"].model == "t-tech/T-lite-it-2.1:q4_K_M"
    assert STACKS["B"].model == "hf.co/ai-sage/GigaChat3.1-10B-A1.8B-GGUF:latest"


def test_safe_diff_rejects_paragraph_rewrite():
    assert diff_candidates("одна строка.\nвторая строка.", "одна строка. вторая строка.", "model-draft") == []


def test_safe_diff_represents_punctuation_insertion_as_bounded_edit():
    edits = diff_candidates("требует корректировки", "требует, корректировки", "model-draft")
    assert edits
    engine = DecisionEngine(min_confidence=0.5, max_changes=4, guard=GenerativeGuard())
    corrected, accepted = engine.apply("требует корректировки", edits)
    assert corrected == "требует, корректировки"
    assert accepted


def test_model_wholesale_rewrite_is_rejected():
    source = "Изучена управленческая роль, должностного лиц в организации."
    corrected = "В документе изучена управленческая роль должностных лиц организации."
    assert diff_candidates(source, corrected, "model-draft") == []


def test_source_text_regression_has_no_structure_change():
    corrected = (
        REGRESSION
        .replace("служебная", "служебной")
        .replace("учебных год", "учебный год")
        .replace("по одна и той", "по одной и той")
    )
    edits = diff_candidates(REGRESSION, corrected, "model-draft")
    assert edits
    assert all("\n" not in c.before and "\n" not in c.after for c in edits)


def test_safe_diff_keeps_grammar_fix_readable():
    """Правка выравнивается по границам слов, а не по символам.

    В v8 такой diff давал `«а» → «ов»`: фрагмент неуникален, правка
    отбрасывалась `DecisionEngine`, а в CHANGES попадал нечитаемый пункт.
    """
    edits = diff_candidates(
        "сотрудником нескольких вида оружия",
        "сотрудником нескольких видов оружия",
        "draft_tlite",
    )
    assert [(c.before, c.after) for c in edits] == [("вида", "видов")]
