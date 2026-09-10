import asyncio

from decision_engine import DecisionEngine, EditCandidate
from hybrid_editor import HybridRouter, STACKS
from local_rules import LocalRuleEngine
from ollama_gec import OllamaGecSpecialist, SYSTEM_PROMPT
from safe_diff import diff_candidates


TEXT = (
    "неправильно определяется оценка за стрельбу при выполнении сотрудником "
    "несколького упражнений стрельб, из одного или нескольких вида оружия "
    "(вооружения) (характерно для всех отделов охраны);"
)

EXPECTED = (
    "неправильно определяется оценка за стрельбу при выполнении сотрудником "
    "нескольких упражнений стрельб, из одного или нескольких видов оружия "
    "(вооружения) (характерно для всех отделов охраны);"
)


def test_v8_stack_contract():
    assert set(STACKS) == {"A", "B", "X", "Y"}
    assert "grammar-first" in STACKS["A"].description
    assert STACKS["X"].model == "qwen3.5-gec+SAGE"


def test_specialist_prompt_is_grammar_focused_and_structure_preserving():
    assert "грамматики" in SYSTEM_PROMPT
    assert "падеж и число" in SYSTEM_PROMPT
    assert "Не перефразируй" in SYSTEM_PROMPT
    specialist = OllamaGecSpecialist()
    assert specialist.num_ctx == 2048
    assert specialist.num_predict == 256


def test_bounded_diff_extracts_the_two_known_local_changes():
    candidates = diff_candidates(TEXT, EXPECTED, "russian-gec", 0.96)
    pairs = {(c.before, c.after) for c in candidates}
    assert ("ого", "их") in pairs
    assert ("а", "ов") in pairs


def test_local_rules_generate_both_target_fixes():
    pairs = {(c.before, c.after) for c in LocalRuleEngine().candidates(TEXT)}
    assert ("несколького упражнений", "нескольких упражнений") in pairs
    assert ("вида", "видов") in pairs


def test_local_rules_apply_target_sentence_without_llm():
    rules = LocalRuleEngine()
    corrected, accepted = DecisionEngine(min_confidence=0.55, max_changes=8).apply(TEXT, rules.candidates(TEXT))
    assert corrected == EXPECTED
    assert {c.category for c in accepted} >= {"rule-quantifier"}


def test_grammar_gate_skips_specialist_when_rule_is_present():
    router = object.__new__(HybridRouter)
    candidates = [EditCandidate("вида", "видов", 0.99, "rule-quantifier", "grammar")]
    assert router._grammar_gate_hit(candidates) is True


def test_specialist_can_be_stubbed_without_network():
    specialist = OllamaGecSpecialist()

    async def fake(_: str) -> str:
        return EXPECTED

    specialist.correct = fake  # type: ignore[method-assign]
    result = asyncio.run(specialist.correct(TEXT))
    assert result == EXPECTED


def test_rank_prefers_russian_gec_over_generic_draft():
    router = object.__new__(HybridRouter)
    strong = EditCandidate("вида", "видов", 0.70, "russian-gec", "specialist")
    weak = EditCandidate("вида", "видов", 0.99, "draft-tlite", "draft")
    ranked = router._rank_candidates([weak, strong])
    assert ranked[0].category == "russian-gec"
