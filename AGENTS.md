# Maincheck harness

Read first. Keep <3 KB. After each repo task update **NOW**, prepend one **LOG** line; keep ≤10.

## Invariants
- Work branch/host checkout: `feat/v20-gec-engine-lab`.
- **Never switch branches** unless user explicitly asks. Check `git branch --show-current`; then `git pull --ff-only origin feat/v20-gec-engine-lab`.
- `feat/v19-rag-sync-shared` is PR base only, not host checkout.
- Only production unit: `ai-suggester.service`; never recreate `v20-main.service`.
- API: `POST /suggest`, multipart files `text` + `context`; preserve cloud contract.
- No phrase-specific patches: fix morphology/syntax/safety classes + regressions.
- Claim quality only after target `scripts/v20/benchmark-current.sh`.

## NOW — 2026-09-17
- PR #50 merged/deployed; host remains on `feat/v20-gec-engine-lab`.
- Target deep-40 v2: **39/40**; recall .967; clean 1.0; FP 0; exact .975; median 626 ms; p95 8375 ms.
- All user cases: 19/19. Only failure: `deep-soglasno-vopreki`: `согласно приказа` not corrected; `вопреки требований` corrected by LT.
- Unit `test_soglasno` passes, so remaining defect is production composition/filtering, not isolated morphology.
- Transformers warning is absent in latest logs: GenerationConfig fix confirmed.
- Tail latency source: RAG lookup blocks ~14–15 s before `RAG evidence`; SAGE/LT deltas stay ~0.4–0.6 s. Next: bounded RAG query timeout/circuit breaker.
- Next code: final deterministic closure after upstream merge; end-to-end combined-frame regression; bounded RAG retrieval. Open a new PR; do not reuse merged #50.

## Active files
- Core: `server/v20/main_app.py`, `server/local/decision_app_v12.py`, `government_frame_stage.py`, `production_safety.py`, `contextual_agreement.py`, `legal_style_stage.py`, `russian_quality_models.py`.
- Tests: `server/local/test_v20_production.py`, `scripts/v20/build_deep_corpus_v2.py`, `scripts/v20/benchmark-current.sh`, `.github/workflows/v20-engine-lab.yml`.
- Service: `server/local/ai-suggester.service`.

## Host commands
```bash
cd /home/service/llama
git branch --show-current
git pull --ff-only origin feat/v20-gec-engine-lab
sudo bash scripts/v20/finalize-main.sh
sudo bash scripts/v20/benchmark-current.sh
```

## LOG (newest first)
- 2026-09-17: target after #50 reached 39/40; final miss is production composition; RAG caused 14–15 s outliers; Transformers warning gone.
- 2026-09-17: #50 fixed dative homonyms, `порядок + процесс`, and model GenerationConfig; CI 23/23 green.
- 2026-09-17: target after #49 reached 38/40 from 31/40; clean 1.0, FP 0.
- 2026-09-17: #49 added bracket safety, copular boundaries, variable dative NP, coordinated processes, auxiliary agreement, normalized duplicate removal, deep corpus v2.
