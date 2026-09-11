"""Canonical entry point for presets A, B, X, Y and Z."""
from __future__ import annotations

import os

_requested_preset = os.getenv("LLM_PRESET", "A").strip().upper()
if _requested_preset == "Z":
    os.environ["LLM_PRESET"] = "X"
    os.environ.setdefault("OLLAMA_GEC_ENABLED", "false")
    os.environ.setdefault("LOCAL_RESCUE_MODE", "never")

from decision_app import app, router
from hybrid_editor import StackInfo
from punctuation_pipeline import LayoutAwareReasoningCascade, OfficePunctuationRules
from v10_rules import V10RuleExtension

_v10 = V10RuleExtension(router.rules.morph_helper)
_punctuation = OfficePunctuationRules(router.rules.morph_helper)
_base_rule_candidates = router.rules.candidates


def _rules_with_extensions(text: str):
    return (
        _base_rule_candidates(text)
        + _v10.candidates(text)
        + _punctuation.candidates(text)
    )


router.rules.candidates = _rules_with_extensions  # type: ignore[method-assign]

if _requested_preset == "Z":
    _cascade = LayoutAwareReasoningCascade(router.client)
    _base_candidates = router.candidates

    async def _reasoning_candidates(text: str, context: str = ""):
        base = await _base_candidates(text, context)
        reasoned = await _cascade.candidates(text, context)
        return router.arbiter.merge(base + reasoned)

    router.candidates = _reasoning_candidates  # type: ignore[method-assign]
    router.info = StackInfo(
        "Z",
        "deterministic + SAGE spelling/punctuation + DeepSeek-R1 7B reasoning + Qwen3.5 extraction",
        _cascade.reasoner,
        True,
    )
    router.ollama_required = lambda: True  # type: ignore[method-assign]
    _base_metrics = router.metrics

    def _metrics_with_reasoning():
        result = _base_metrics()
        result["reasoning_cascade"] = _cascade.metrics().__dict__
        return result

    router.metrics = _metrics_with_reasoning  # type: ignore[method-assign]
