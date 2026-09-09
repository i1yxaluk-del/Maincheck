import asyncio

from decision_engine import EditCandidate
from hybrid_editor import HybridRouter, STACKS
from russian_mlm_corrector import RussianMlmCorrector
from safe_diff import diff_candidates


TEXT = (
    "неправильно определяется оценка за стрельбу при выполнении сотрудником несколького "
    "упражнений стрельб, из одного или нескольких вида оружия (вооружения)"
)


def test_v6_stack_contract_is_local_first():
    assert set(STACKS) == {"A", "B", "X", "Y"}
    assert "Russian MLM" in STACKS["A"].description or "ruBert" in STACKS["A"].description
    assert "SAGE" in STACKS["X"].description


def test_v6_mlm_defaults_are_lightweight_and_local():
    model = RussianMlmCorrector()
    assert model.model_id == "ai-forever/ruBert-base"
    assert model.enabled is True


def test_mlm_stub_can_supply_targeted_word_edits_without_rewriting():
    model = RussianMlmCorrector()

    async def fake_candidates(_text: str):
        return [
            EditCandidate("несколького", "нескольких", 0.88, "russian-mlm", "stub"),
            EditCandidate("вида", "видов", 0.86, "russian-mlm", "stub"),
        ]

    model.candidates = fake_candidates  # type: ignore[method-assign]
    result = asyncio.run(model.candidates(TEXT))
    assert {(c.before, c.after) for c in result} == {
        ("несколького", "нескольких"),
        ("вида", "видов"),
    }


def test_adaptive_router_can_prefers_local_signal_over_rescue():
    router = object.__new__(HybridRouter)
    strong = [EditCandidate("вида", "видов", 0.86, "russian-mlm", "MLM")]
    # With the default gate of 2, one candidate deliberately does not count as sufficient signal.
    assert router._has_sufficient_local_signal(strong) is False
    stronger = strong + [EditCandidate("несколького", "нескольких", 0.89, "rule-quantifier", "rule")]
    assert router._has_sufficient_local_signal(stronger) is True


def test_safe_diff_rejects_sentence_reordering_even_when_lengths_match():
    source = "Оценка за стрельбу определяется при выполнении сотрудником упражнений."
    reordered = "При выполнении сотрудником упражнений оценка за стрельбу определяется."
    assert diff_candidates(source, reordered, "draft-tlite") == []


def test_safe_diff_still_allows_small_grammar_fix():
    source = "сотрудником нескольких вида оружия"
    corrected = "сотрудником нескольких видов оружия"
    edits = diff_candidates(source, corrected, "draft-tlite")
    assert edits
    assert any("вида" in c.before and "видов" in c.after for c in edits)
