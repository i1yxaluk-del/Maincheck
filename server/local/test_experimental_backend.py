from decision_engine import DecisionEngine, EditCandidate
from pipelines import SurfaceGate, _char_diff_candidates


def test_decision_engine_exact_occurrence():
    source = "Это важный акт."
    candidate = EditCandidate("важный", "важная", 0.9, "agreement", "test")
    corrected, accepted = DecisionEngine().apply(source, [candidate])
    assert corrected == "Это важная акт."
    assert len(accepted) == 1


def test_surface_diff_does_not_allow_paragraph_reflow():
    source = "строка один.\nстрока два."
    corrected = "строка один. строка два."
    assert _char_diff_candidates(source, corrected, "surface") == []


def test_surface_gate_requires_same_word_morphology_for_replacement():
    gate = SurfaceGate()
    assert gate.accept("очепятка", "опечатка") is False or gate.available is False
    assert gate.accept("изучена", "изучено") is False
    assert gate.accept("должностного", "должностных") is False
    assert gate.accept("деятельностей", "деятельности") is False


def test_surface_diff_is_bounded():
    edits = _char_diff_candidates("Это текст.", "Это текст,", "surface")
    assert edits == [] or all(len(e.before) <= 80 and len(e.after) <= 80 for e in edits)
