import asyncio

from hybrid_editor import STACKS
from russian_quality_models import SageRussianCorrector
from safe_diff import diff_candidates


TEXT = "привет как дила, что делаеш севодня"
CORRECTED = "Привет, как дела, что делаешь сегодня?"


def test_v4_stacks_use_real_candidate_cascade():
    assert set(STACKS) == {"A", "B", "X", "Y"}
    assert "SAGE" in STACKS["A"].description
    assert "multi-candidate" in STACKS["X"].description


def test_sage_is_fast_specialist_by_default():
    model = SageRussianCorrector()
    assert model.enabled is True
    assert model.model_id == "ai-forever/sage-fredt5-distilled-95m"


def test_bounded_diff_turns_full_correction_into_local_edits():
    candidates = diff_candidates(TEXT, CORRECTED, "test", 0.9)
    pairs = {(c.before, c.after) for c in candidates}
    assert ("привет", "Привет") in pairs
    assert any("ди" in before or "делаеш" in before for before, _ in pairs)


def test_sage_corrector_can_be_stubbed_without_model_download():
    model = SageRussianCorrector()

    def fake(text: str) -> str:
        return text.replace("дила", "дела").replace("делаеш", "делаешь")

    model._correct_sync = fake  # type: ignore[method-assign]
    result = asyncio.run(model.correct("как дила, что делаеш"))
    assert result == "как дела, что делаешь"
