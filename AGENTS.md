# Maincheck harness

Read first. Keep this file compact (<3 KB). After each repo task update **NOW** and prepend one line to **LOG**; keep at most 10 log lines.

## Invariants
- Work branch/host checkout: `feat/v20-gec-engine-lab`.
- **Do not switch branches** unless the user explicitly asks. First check `git branch --show-current`; if already on work branch, use only `git pull --ff-only origin feat/v20-gec-engine-lab`.
- `feat/v19-rag-sync-shared` is PR base only, not the host checkout.
- Production has one unit: `ai-suggester.service`; never recreate `v20-main.service`.
- Local/cloud endpoint: `POST /suggest`, multipart files `text` + `context`; preserve cloud contract.
- Avoid phrase-specific patches; implement morphology/syntax/safety classes and add regression tests.
- Never claim full quality before target-host `scripts/v20/benchmark-current.sh`.

## NOW — 2026-09-17
- PR: #50 open, head `622f368bc6f2125bce49727b37bcb11e4b907e78`, CI 23/23 green before this doc-only commit.
- PR #49 merged/deployed. Target deep-40 v2: 38/40; recall .933; clean 1.0; FP 0; exact .95; median 617 ms; p95 2948 ms.
- Remaining target failures addressed in #50: `deep-soglasno-vopreki`, `deep-style-and-government`.
- PR #50 also fixes inherited Transformers `max_length=256`; verify warning disappears on target.
- Next: merge/deploy #50 from the existing work branch, restart via `sudo bash scripts/v20/finalize-main.sh`, rerun benchmark; do not switch branches.

## Active files
- Core: `server/local/government_frame_stage.py`, `production_safety.py`, `contextual_agreement.py`, `legal_style_stage.py`, `russian_quality_models.py`.
- Tests: `server/local/test_v20_production.py`, `scripts/v20/build_deep_corpus_v2.py`, `scripts/v20/benchmark-current.sh`, `.github/workflows/v20-engine-lab.yml`.
- Service: `server/v20/main_app.py`, `server/local/ai-suggester.service`.

## Host commands
```bash
cd /home/service/llama
git branch --show-current
git pull --ff-only origin feat/v20-gec-engine-lab
sudo bash scripts/v20/finalize-main.sh
sudo bash scripts/v20/benchmark-current.sh
```

## LOG (newest first)
- 2026-09-17: #50 fixed genitive→dative homonym selection, `порядок + процесс`, and model GenerationConfig; isolated CI regressions; 23/23 green.
- 2026-09-17: target after #49 reached 38/40 from 31/40; clean preservation rose to 1.0 and FP fell to 0.
- 2026-09-17: #49 added context-aware bracket safety, copular boundaries, variable dative NP, coordinated processes, auxiliary agreement, normalized duplicate removal, deep corpus v2.
