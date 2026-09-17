# Maincheck harness

Read first. Keep <3 KB. After each repo task update **NOW**, prepend one **LOG** line; keep ≤10.

## Invariants
- Work branch/host checkout: `feat/v20-gec-engine-lab`.
- **Never switch branches** unless user explicitly asks. Check `git branch --show-current`; then `git pull --ff-only origin feat/v20-gec-engine-lab`.
- `feat/v19-rag-sync-shared` is PR base only, not host checkout.
- Only production unit: `ai-suggester.service`; never recreate `v20-main.service`.
- API: `POST /suggest`, multipart files `text` + `context`; preserve cloud contract.
- No phrase-specific patches: fix morphology/syntax/safety classes + regressions.
- Claim quality only after target `scripts/v20/benchmark-current.sh` and real-document replay.

## NOW — 2026-09-17
- PR #53 open: https://github.com/i1yxaluk-del/Maincheck/pull/53; head `6855fd76049cc93ba10c91480784a893ab15da2b`; all 24 current CI jobs green.
- Real incident: correct `два государственных контракта` was damaged into `два ггосударственногоконтракта`; `поставщиком, Центром` became `ппоставщикомЦентром`.
- #53 adds generative word-boundary/duplicate/capital-concatenation guards, same-sentence family taint, and valid 2–4 numeral-government protection.
- #53 also reintroduces deterministic legal/government candidates at the final production boundary, targeting the remaining `согласно приказа` loss.
- Baseline target deep-40: 39/40; recall .967; clean 1.0; FP 0; exact .975; median 589 ms; p95 1028 ms.
- One 14.8 s RAG outlier remains outside p95; handle retrieval timeout separately after quality validation.
- Next: merge #53, pull existing branch, finalize, run deep-40 and replay the full reported paragraph. Do not claim fixed before target output.

## Active files
- `server/local/production_safety.py`, `server/v20/main_app.py`, `server/local/test_v20_real_document_safety.py`, `.github/workflows/v20-engine-lab.yml`.
- Core context: `decision_app_v12.py`, `government_frame_stage.py`, `safe_diff.py`, `decision_engine.py`.

## Host commands
```bash
cd /home/service/llama
git branch --show-current
git pull --ff-only origin feat/v20-gec-engine-lab
sudo bash scripts/v20/finalize-main.sh
sudo bash scripts/v20/benchmark-current.sh
```

## LOG (newest first)
- 2026-09-17: opened #53; structural generative guards + numeral protection + deterministic final closure; CI 24/24 green.
- 2026-09-17: real paragraph exposed accepted SAGE word concatenation/duplication and false agreement after `два` despite 39/40 benchmark.
- 2026-09-17: target after #50 stayed 39/40; median 589 ms, p95 1028 ms, one hidden 14.8 s RAG outlier.
- 2026-09-17: #50 fixed dative homonyms, `порядок + процесс`, and model GenerationConfig; CI green.
