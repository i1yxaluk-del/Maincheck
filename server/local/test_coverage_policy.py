from decision_engine import EditCandidate
from coverage_policy import (
    is_punctuation_only,
    needs_deep_review,
    needs_rescue_despite_verified_punctuation,
)


def edit(before, after, category="sage-spell-punc"):
    return EditCandidate(before, after, 0.8, category)


def test_one_comma_no_longer_suppresses_grammar_review(monkeypatch):
    monkeypatch.setenv("REASONING_COVERAGE_MIN_CHARS", "20")
    text = "Выявлены нарушения, порядка ведений учета учебного гранатометания."
    assert needs_deep_review(text, [edit("нарушения,", "нарушения")])


def test_substantive_fast_fix_can_finish_without_reasoner(monkeypatch):
    monkeypatch.setenv("REASONING_COVERAGE_MIN_CHARS", "20")
    text = "Выявлены нарушения порядка ведений учета учебного гранатометания."
    assert not needs_deep_review(
        text, [edit("ведений", "ведения", "rule-process-government")]
    )


def test_verified_punctuation_does_not_block_rescue():
    candidates = [edit("Центра,", "Центра", "rule-predicate-boundary")]
    assert is_punctuation_only(candidates[0])
    assert needs_rescue_despite_verified_punctuation(candidates, False)


def test_original_rescue_decision_is_preserved_for_word_edits():
    candidates = [edit("подготовок", "подготовке", "rule-agreement")]
    assert not needs_rescue_despite_verified_punctuation(candidates, False)
    assert needs_rescue_despite_verified_punctuation(candidates, True)
