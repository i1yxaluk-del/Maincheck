from decision_engine import DecisionEngine
from hybrid_editor import STACKS, diff_candidates
from safe_diff import diff_candidates as safe_diff_candidates


REGRESSION = (
    "План профессиональной служебная и физической подготовки на 2026 учебных "
    "год требует, корректировки (штабная тренировка спланирована в один день "
    "с тактико-специальным занятием с элементами командно-штабной тренировки "
    "по одна и той же теме);"
)


def test_new_preset_contract():
    assert set(STACKS) == {"A", "B", "X", "Y"}
    assert STACKS["C"].model == "" if "C" in STACKS else True


def test_safe_diff_rejects_paragraph_rewrite():
    assert safe_diff_candidates("одна строка.\nвторая строка.", "одна строка. вторая строка.", "model-draft") == []


def test_safe_diff_represents_punctuation_insertion_as_bounded_edit():
    edits = safe_diff_candidates("требует корректировки", "требует, корректировки", "model-draft")
    assert edits
    engine = DecisionEngine(min_confidence=0.5, max_changes=4)
    corrected, accepted = engine.apply("требует корректировки", edits)
    assert corrected == "требует, корректировки"
    assert accepted


def test_model_wholesale_rewrite_is_rejected():
    source = "Изучена управленческая роль, должностного лиц в организации."
    corrected = "В документе изучена управленческая роль должностных лиц организации."
    assert diff_candidates(source, corrected, "model-draft") == []


def test_source_text_regression_has_no_structure_change():
    corrected = REGRESSION.replace("служебная", "служебной").replace("учебных год", "учебный год").replace("по одна и той", "по одной и той")
    edits = safe_diff_candidates(REGRESSION, corrected, "model-draft")
    assert edits
    assert all("\n" not in c.before and "\n" not in c.after for c in edits)
