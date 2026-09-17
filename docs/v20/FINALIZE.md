# Final production deployment

The target-hardware comparison selected the guarded `main` pipeline. OpenVINO was fast but changed dates and official abbreviations; llama-json preserved clean text but corrected none of the positive cases.

Production now uses one API unit only: `ai-suggester.service`. Its entrypoint is `v20.main_app:app`, and that entrypoint forces stack Z instead of inheriting the obsolete `LLM_PRESET=A` from `.env`.

The main pipeline combines punctuation, spelling, morphology, normative RAG, bounded reasoning and deterministic legal-style checks. Legal checks include adjacent repeated phrases, audit-context government (`в отношениях Центра` → `в отношении Центра`) and stable official collocations. Generative edits cannot change digits, acronyms, quoted names or numbered-list wording without trusted evidence.

## Finalize

```bash
sudo /home/service/llama/scripts/v20/finalize-main.sh
```

The script removes all `v20-*` runtime units and experimental resources, cleans optional GPU/OpenVINO packages, installs the production unit as `/etc/systemd/system/ai-suggester.service`, starts Ollama, requires a non-degraded health response containing `stack=Z`, and records freed bytes.

Success marker:

```text
V20 FINALIZED service=ai-suggester.service stack=Z freed_bytes=<bytes> health=OK ...
```

RAG data, user dictionaries, benchmark results and a one-time backup of the previous unit are preserved. To skip shared-venv cleanup, set `V20_PURGE_SHARED_EXPERIMENT_DEPS=false`.
