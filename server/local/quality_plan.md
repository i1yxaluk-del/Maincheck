# Local GEC quality plan v2.4

The service is a local editor of minimal corrections, not a text rewriter.

## Production A
`normalize -> deterministic morphology rescue -> sparse/trigram example retrieval -> T-lite local edits -> deterministic validation -> exact application`

## Experimental G
`detector + morphology rescue -> T-lite verifier -> DecisionEngine`

## Experimental F
`Spell-Corrector-RU-4B -> bounded diff -> structure/morphology gate -> DecisionEngine`

Recent Russian GEC research supports edit-based correction and targeted example retrieval. LORuGEC reports gains from GECToR-based example selection, while BEA 2026 shows that aggregate F0.5 can hide severe per-rule regressions. Evaluation therefore tracks rule-level precision/recall, destructive edits and latency in addition to aggregate metrics.
