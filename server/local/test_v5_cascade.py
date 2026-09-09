import asyncio

from decision_engine import EditCandidate
from hybrid_editor import HybridRouter, STACKS
from russian_gec_backend import RussianGecSpecialist
from russian_quality_models import SageRussianCorrector
from safe_diff import diff_candidates


TEXT = "неправильно определяется оценка за стрельбу при выполнении сотрудником несколького упражнений стрельб, из одного или нескольких вида оружия"


def test_v5_stack_contract():
    assert set(STACKS) == {"A", "B", "X", "Y"}
    assert "SyntErr" in STACKS["A"].description
    assert STACKS["A"].model == "t-tech/T-lite-it-2.1:q4_K_M"
    assert STACKS["B"].model == "hf.co/ai-sage/GigaChat3.1-10B-A1.8B-GGUF:latest"


def test_fast_models_are_configured_for_real_specialist_roles():
    sage = SageRussianCorrector()
    gec = RussianGecSpecialist()
    assert sage.model_id == "ai-forever/sage-fredt5-distilled-95m"
    assert gec.base_model == "Qwen/Qwen3.5-0.8B"
    assert gec.adapter_repo == "synterr-nlp/bea2026-gec-adapters"
    assert gec.adapter_subfolder == "v4_qwen35_08b_lorugec"


def test_bounded_diff_extracts_multiple_error_sites():
    corrected = TEXT.replace("несколького", "нескольких").replace("вида оружия", "видов оружия")
    candidates = diff_candidates(TEXT, corrected, "russian-gec", 0.93)
    pairs = {(c.before, c.after) for c in candidates}
    assert any("несколького" in before and "нескольких" in after for before, after in pairs)
    assert any("вида" in before and "видов" in after for before, after in pairs)


def test_router_ranks_specialists_above_generic_draft():
    router = object.__new__(HybridRouter)
    strong = EditCandidate("вида", "видов", 0.80, "russian-gec", "GEC")
    weak = EditCandidate("вида", "видов", 0.99, "draft-tlite", "draft")
    ranked = router._rank_candidates([weak, strong])
    assert ranked[0].category == "russian-gec"


def test_specialist_backend_can_be_stubbed_without_download():
    backend = RussianGecSpecialist()

    def fake(text: str) -> str:
        return text.replace("несколького", "нескольких").replace("вида", "видов")

    backend._correct_sync = fake  # type: ignore[method-assign]
    result = asyncio.run(backend.correct(TEXT))
    assert "нескольких упражнений" in result
    assert "нескольких видов оружия" in result
