from decision_engine import EditCandidate
from experimental_backend import _parse_model_json, _safe_diff_candidates


def test_parse_model_json_roundtrip():
    raw = '{"edits":[{"before":"ночных наряда","after":"ночных нарядов","confidence":0.96,"category":"agreement","reason":"согласование"}]}'
    edits = _parse_model_json(raw)
    assert len(edits) == 1
    assert edits[0].before == "ночных наряда"
    assert edits[0].after == "ночных нарядов"


def test_safe_diff_rejects_lexical_insertion():
    source = "рабочего времени"
    corrected = "рабочегочасовымирабочего времени"
    edits = _safe_diff_candidates(source, corrected, "specialized-gec")
    assert edits == []


def test_safe_diff_keeps_local_spelling_change():
    source = "Это очепятка."
    corrected = "Это опечатка."
    edits = _safe_diff_candidates(source, corrected, "spelling")
    assert edits
    assert all(isinstance(edit, EditCandidate) for edit in edits)
