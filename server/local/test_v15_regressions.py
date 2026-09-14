from decision_engine import DecisionEngine, EditCandidate
from client_safe_edits import materialize_client_safe_deletions
from morphology import get_morphology
from v15_rules import V15RuleExtension


def apply_rules(text: str):
    candidates = V15RuleExtension(get_morphology()).candidates(text)
    return DecisionEngine(min_confidence=0.55).apply(text, candidates)


def test_full_report_example_gets_two_local_edits():
    text = ("План\nпрофессиональной служебной, и физической\nподготовки\n"
            "на 2026 учебный год требует\nкорректировки (штабная тренировка\n"
            "спланирована в один день с тактико-специальным\nзанятиями\n"
            "с элементами командно-штабной тренировки\nпо одной и той же теме);")
    corrected, accepted = apply_rules(text)
    assert "служебной и физической" in corrected.replace("\n", " ")
    assert "тактико-специальными\nзанятиями" in corrected
    assert len(accepted) == 2
    assert all(c.before and c.after for c in accepted)


def test_compound_agreement_is_lexically_general():
    corrected, _ = apply_rules("Отчет дополнен организационно-техническим мероприятиями.")
    assert corrected == "Отчет дополнен организационно-техническими мероприятиями."


def test_correct_compound_modifier_is_preserved():
    text = "Проведены командно-штабные тренировки."
    corrected, accepted = apply_rules(text)
    assert corrected == text
    assert not accepted


def test_comma_before_single_coordinator_is_removed():
    corrected, _ = apply_rules("Изучены служебная, и физическая подготовка.")
    assert corrected == "Изучены служебная и физическая подготовка."


def test_punctuation_deletion_is_anchored_for_writer_client():
    text = "служебной, и физической подготовки"
    raw = [EditCandidate(",", "", 0.8, "sage-rupunct", start=9)]
    materialized = materialize_client_safe_deletions(text, raw)
    assert len(materialized) == 1
    assert materialized[0].before == "служебной,"
    assert materialized[0].after == "служебной"
    corrected, accepted = DecisionEngine(min_confidence=0.55).apply(text, materialized)
    assert corrected == "служебной и физической подготовки"
    assert accepted


def test_non_deletion_candidate_is_unchanged():
    candidate = EditCandidate("ошбка", "ошибка", 0.9, "dict-spell", start=0)
    assert materialize_client_safe_deletions("ошбка", [candidate]) == [candidate]
