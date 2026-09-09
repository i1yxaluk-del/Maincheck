import asyncio

from decision_engine import EditCandidate
from hybrid_editor import HybridRouter, STACKS
from ollama_gec import OllamaGecSpecialist, SYSTEM_PROMPT
from safe_diff import diff_candidates


TEXT = (
    "неправильно определяется оценка за стрельбу при выполнении сотрудником "
    "несколького упражнений стрельб, из одного или нескольких вида оружия "
    "(вооружения) (характерно для всех отделов охраны);"
)


def test_v7_stack_contract():
    assert set(STACKS) == {"A", "B", "X", "Y"}
    assert "Qwen3.5 GEC" in STACKS["A"].description
    assert STACKS["X"].model == "qwen3.5-gec+SAGE"


def test_specialist_uses_exact_golden_prompt_and_greedy_settings():
    assert "Сохрани язык, слова и смысл" in SYSTEM_PROMPT
    specialist = OllamaGecSpecialist()
    assert specialist.num_ctx == 2048
    assert specialist.num_predict == 256


def test_bounded_diff_extracts_the_two_known_grammar_edits_without_rewrite():
    corrected = TEXT.replace("несколького", "нескольких").replace("нескольких вида", "нескольких видов")
    candidates = diff_candidates(TEXT, corrected, "russian-gec", 0.96)
    pairs = {(c.before, c.after) for c in candidates}
    assert any("несколького" in before and "нескольких" in after for before, after in pairs)
    assert any("вида" in before and "видов" in after for before, after in pairs)


def test_specialist_can_be_stubbed_without_network():
    specialist = OllamaGecSpecialist()

    async def fake(_: str) -> str:
        return TEXT.replace("несколького", "нескольких").replace("нескольких вида", "нескольких видов")

    specialist.correct = fake  # type: ignore[method-assign]
    result = asyncio.run(specialist.correct(TEXT))
    assert "нескольких упражнений" in result
    assert "нескольких видов оружия" in result


def test_rank_prefers_russian_gec_over_generic_draft():
    router = object.__new__(HybridRouter)
    strong = EditCandidate("вида", "видов", 0.70, "russian-gec", "specialist")
    weak = EditCandidate("вида", "видов", 0.99, "draft-tlite", "draft")
    ranked = router._rank_candidates([weak, strong])
    assert ranked[0].category == "russian-gec"
