import asyncio

from decision_engine import DecisionEngine
from hybrid_editor import HybridRouter
from local_rules import LocalRuleEngine


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


def test_reference_text_is_fully_corrected_without_llm():
    candidates = LocalRuleEngine().candidates(TEXT)
    corrected, accepted = DecisionEngine(min_confidence=0.55, max_changes=8).apply(TEXT, candidates)
    assert corrected == EXPECTED
    assert len(accepted) == 2


def test_grammar_gate_is_decision_boundary():
    router = object.__new__(HybridRouter)
    candidates = LocalRuleEngine().candidates(TEXT)
    assert router._grammar_gate_hit(candidates)


def test_gate_helper_is_async_safe():
    async def check() -> bool:
        router = object.__new__(HybridRouter)
        return router._grammar_gate_hit(LocalRuleEngine().candidates(TEXT))

    assert asyncio.run(check()) is True
