from decision_engine import DecisionEngine, EditCandidate
from pipelines import SurfaceGate, surface_candidates


BAD_F_SOURCE = """Изучена
управленческая роль, должностного
лиц в организации служебно-боевой
деятельностей
и фактическое положение дел

в
подразделениях Центра."""

BAD_F_OUTPUT = """Изучено
управленческая роль, должностных лица
в организации служебно-боевой деятельности
и фактическое положение дел

в
подразделениях Центра."""


def test_decision_engine_exact_occurrence():
    source = "Это важный акт."
    candidate = EditCandidate("важный", "важная", 0.9, "agreement", "test")
    corrected, accepted = DecisionEngine().apply(source, [candidate])
    assert corrected == "Это важная акт."
    assert len(accepted) == 1


def test_surface_diff_rejects_paragraph_reflow():
    source = "строка один.\nстрока два."
    corrected = "строка один. строка два."
    assert surface_candidates(source, corrected) == []


def test_surface_gate_rejects_observed_f_form_changes():
    gate = SurfaceGate()
    assert gate.accept("изучена", "изучено") is False
    assert gate.accept("должностного", "должностных") is False
    assert gate.accept("деятельностей", "деятельности") is False


def test_full_observed_f_output_has_no_admissible_surface_edits():
    gate = SurfaceGate()
    candidates = surface_candidates(BAD_F_SOURCE, BAD_F_OUTPUT)
    safe = [c for c in candidates if gate.accept(c.before, c.after)]
    assert safe == []


def test_surface_changes_are_bounded():
    edits = surface_candidates("Это текст.", "Это текст,")
    assert all(len(e.before) <= 80 and len(e.after) <= 80 for e in edits)
