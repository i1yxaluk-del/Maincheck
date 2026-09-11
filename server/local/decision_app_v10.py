"""v10 entry point: v9 service plus post-rollout high-precision rules."""
from __future__ import annotations

from decision_app import app, router  # re-exported for uvicorn
from v10_rules import V10RuleExtension

_extension = V10RuleExtension(router.rules.morph_helper)
_v9_candidates = router.rules.candidates


def _candidates_with_v10(text: str):
    return _v9_candidates(text) + _extension.candidates(text)


router.rules.candidates = _candidates_with_v10  # type: ignore[method-assign]
