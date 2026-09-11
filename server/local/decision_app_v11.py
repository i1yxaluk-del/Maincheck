"""Canonical entry point for presets A, B, X, Y and Z."""
from __future__ import annotations

import asyncio
import os

_requested_preset = os.getenv("LLM_PRESET", "A").strip().upper()
if _requested_preset == "Z":
    os.environ["LLM_PRESET"] = "X"
    os.environ["OLLAMA_GEC_ENABLED"] = "false"
    os.environ["LOCAL_RESCUE_MODE"] = "never"

from decision_app import app, router
from hybrid_editor import StackInfo
from punctuation_pipeline import (
    LayoutAwareReasoningCascade,
    OfficePunctuationRules,
    StructuralPunctuationRules,
)
from rupunct_stage import RuPunctStage
from v10_rules import V10RuleExtension

_v10 = V10RuleExtension(router.rules.morph_helper)
_office_punctuation = OfficePunctuationRules(router.rules.morph_helper)
_structural_punctuation = StructuralPunctuationRules(router.rules.morph_helper)
_rupunct = RuPunctStage()
_base_rule_candidates = router.rules.candidates


def _rules_with_extensions(text: str):
    return (
        _base_rule_candidates(text)
        + _v10.candidates(text)
        + _office_punctuation.candidates(text)
        + _structural_punctuation.candidates(text)
    )


router.rules.candidates = _rules_with_extensions  # type: ignore[method-assign]
_base_candidates = router.candidates

async def _candidates_with_rupunct(text: str, context: str = ""):
    base, punctuation = await asyncio.gather(
        _base_candidates(text, context), _rupunct.candidates(text),
    )
    return router.arbiter.merge(base + punctuation)

router.candidates = _candidates_with_rupunct  # type: ignore[method-assign]

if _requested_preset == "Z":
    _cascade = LayoutAwareReasoningCascade(router.client)
    _fast_candidates = router.candidates

    async def _reasoning_candidates(text: str, context: str = ""):
        fast = await _fast_candidates(text, context)
        mode = os.getenv("REASONING_MODE", "fallback").strip().lower()
        if mode == "never" or (mode != "always" and fast):
            return fast
        reasoned = await _cascade.candidates(text, context)
        return router.arbiter.merge(fast + reasoned)

    router.candidates = _reasoning_candidates  # type: ignore[method-assign]
    router.info = StackInfo(
        "Z", "fast punctuation ensemble + fallback DeepSeek-R1 7B",
        _cascade.reasoner, True,
    )
    router.ollama_required = lambda: True  # type: ignore[method-assign]
else:
    _cascade = None

_base_metrics = router.metrics
def _metrics_with_specialists():
    result = _base_metrics()
    result["rupunct"] = _rupunct.metrics().__dict__
    if _cascade is not None:
        result["reasoning_cascade"] = _cascade.metrics().__dict__
    return result
router.metrics = _metrics_with_specialists  # type: ignore[method-assign]
