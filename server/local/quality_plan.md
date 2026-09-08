# Local GEC quality plan v2.4

## Goal

This service is a **local editor of minimal corrections**, not a text-rewriter. The system should improve precision and useful recall while keeping paragraph structure, terminology and valid word forms intact.

## Research-backed design decisions

1. Russian GEC benefits from edit-based methods and targeted example retrieval. The 2025 RussianGEC sequence-tagger work reports strong results on RU-Lang8/GERA, while the LORuGEC work reports that GECToR-based example selection improves few-shot LLM correction. The 2026 BEA diagnostic work further shows that aggregate F0.5 can hide rule-level regressions, so this project must track errors by rule rather than only total score.
2. Large full-text rewriting models are not a suitable primary editor for the current CPU host: inference latency and uncontrolled inflection changes are too risky. Full-text generation remains an isolated experiment only.
3. The local stack therefore uses a staged architecture: deterministic high-precision candidate detection, targeted examples, one structured T-lite pass, then deterministic validation and exact application.

## Production stack A

`normalize -> deterministic morphology rescue -> sparse/trigram few-shot retrieval -> T-lite local-edit generation -> merge -> DecisionEngine`

The T-lite prompt never asks for a rewritten paragraph. The output is structured local edits only.

## Experimental stack G

`detector -> T-lite verifier -> DecisionEngine`

No generation of new candidates by the verifier. This is the conservative/high-precision experiment.

## Experimental stack F

`Spell-Corrector-RU-4B full-text generation -> bounded diff -> morphology gate -> DecisionEngine`

F is not production and is fail-closed on morphology/structure changes.

## Evaluation policy

Every release must contain regression cases for:

- agreement: `должностного лиц -> должностного лица`
- forbidden inflection rewrites: `изучена -> изучено`, `должностного -> должностных`, `деятельностей -> деятельности`
- participle + instrumental agent constructions
- punctuation insertion/deletion
- spelling changes
- abbreviations and protected dictionary words
- paragraph and soft-line-break preservation

Track at minimum: precision, recall, F0.5, false-positive rate, destructive edits, median latency, p95 latency and number of accepted edits.
