"""Entry point supporting experimental preset Z.

Z runs deterministic rules first, then a 7B reasoning editor and finally the
existing 0.8B GEC model as a result extractor. Other presets retain v10 behavior.
"""
from __future__ import annotations

import os

_requested_preset = os.getenv("LLM_PRESET", "A").strip().upper()
if _requested_preset == "Z":
    # decision_app knows A/B/X/Y. Import it in deterministic-only X mode and
    # attach the experimental cascade below.
    os.environ["LLM_PRESET"] = "X"
    os.environ.setdefault("SAGE_CORRECTOR_ENABLED", "false")
    os.environ.setdefault("OLLAMA_GEC_ENABLED", "false")
    os.environ.setdefault("LOCAL_RESCUE_MODE", "never")

from decision_app import app, router
from hybrid_editor import StackInfo
from reasoning_cascade import ReasoningCascade
from v10_rules import V10RuleExtension

_v10 = V10RuleExtension(router.rules.morph_helper)
_v9_rule_candidates = router.rules.candidates


def _rules_with_v10(text: str):
    return _v9_rule_candidates(text) + _v10.candidates(text)


router.rules.candidates = _rules_with_v10  # type: ignore[method-assign]

if _requested_preset == "Z":
    _cascade = ReasoningCascade(router.client)
    _deterministic_candidates = router.candidates

    async def _reasoning_candidates(text: str, context: str = ""):
        base = await _deterministic_candidates(text, context)
        reasoned = await _cascade.candidates(text, context)
        return router.arbiter.merge(base + reasoned)

    router.candidates = _reasoning_candidates  # type: ignore[method-assign]
    router.info = StackInfo(
        "Z",
        "experimental: deterministic + DeepSeek-R1 7B reasoning + Qwen3.5 0.8B extraction",
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
