from decision_engine import EditCandidate
from hybrid_editor import HybridRouter, STACKS
from safe_diff import diff_candidates


TEXT = "неправильно определяется оценка за стрельбу при выполнении сотрудником несколького упражнений стрельб, из одного или нескольких вида оружия"


def test_v8_stack_contract_is_preserved_for_compatibility():
    assert set(STACKS) == {"A", "B", "X", "Y"}
    assert "SAGE" in STACKS["A"].description
    assert "Qwen3.5 GEC" in STACKS["A"].description
    assert STACKS["A"].model == "t-tech/T-lite-it-2.1:q4_K_M"
    assert STACKS["B"].model == "hf.co/ai-sage/GigaChat3.1-10B-A1.8B-GGUF:latest"


def test_bounded_diff_extracts_multiple_local_changes():
    corrected = TEXT.replace("несколького", "нескольких").replace("вида оружия", "видов оружия")
    candidates = diff_candidates(TEXT, corrected, "russian-gec", 0.93)
    pairs = {(c.before, c.after) for c in candidates}
    assert ("ого", "их") in pairs
    assert ("а", "ов") in pairs


def test_router_ranks_specialists_above_generic_draft():
    router = object.__new__(HybridRouter)
    strong = EditCandidate("вида", "видов", 0.80, "russian-gec", "specialist")
    weak = EditCandidate("вида", "видов", 0.99, "draft-tlite", "draft")
    ranked = router._rank_candidates([weak, strong])
    assert ranked[0].category == "russian-gec"
