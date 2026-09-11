from __future__ import annotations

import asyncio

from decision_engine import DecisionEngine
from morphology import get_morphology
from punctuation_pipeline import (
    LayoutAwareReasoningCascade,
    OfficePunctuationRules,
    collapse_soft_breaks,
    restore_soft_breaks,
)


def test_soft_wraps_are_collapsed_for_models_and_restored_after_edit():
    source = "проведенные\nзанятия в\nмарте не отражены."
    flat = collapse_soft_breaks(source)
    assert "\n" not in flat
    corrected = "проведенные занятия в марте, не отражены."
    restored = restore_soft_breaks(source, corrected)
    assert restored == "проведенные\nзанятия в\nмарте, не отражены."


def test_reasoner_receives_unwrapped_text_but_diff_targets_original_layout():
    cascade = LayoutAwareReasoningCascade()
    seen: list[str] = []

    async def fake_correct(text, context=""):
        seen.append(text)
        return text.replace("марте не", "марте, не")

    cascade.correct = fake_correct  # type: ignore[method-assign]
    edits = asyncio.run(cascade.candidates("занятия\nв\nмарте не отражены."))
    assert seen == ["занятия в марте не отражены."]
    assert any(e.after == "," or "," in e.after for e in edits)


def test_company_name_comma_is_moved_to_close_participial_phrase():
    text = (
        "проведенные\nс личным составом, ООО\n«Рога и Копыта»\n"
        "ежеквартальные контрольные занятия не отражены."
    )
    candidates = OfficePunctuationRules(get_morphology()).candidates(text)
    corrected, accepted = DecisionEngine(min_confidence=0.5).apply(text, candidates)
    assert corrected == (
        "проведенные\nс личным составом ООО\n«Рога и Копыта»,\n"
        "ежеквартальные контрольные занятия не отражены."
    )
    assert len(accepted) == 2
