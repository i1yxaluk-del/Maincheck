# Final production selection

Target-hardware benchmark selected `main`: it corrected 8/10 positive cases including the long copular-agreement paragraph. OpenVINO was faster but changed dates and official abbreviations; llama-json preserved clean text but corrected 0/10 positive cases.

The production main wrapper now raises the acceptance threshold, shortens the reasoner budget to 12 seconds, skips the expensive reasoner for short clean selections, and rejects changes to digits, acronyms, quoted names and numbered-list wording unless the canonical RAG terminology rule authorizes the acronym.

## Remove experiment resources and leave main running

```bash
sudo /home/service/llama/scripts/v20/finalize-main.sh
```

The script stops experimental and transient units, removes OpenVINO/llama virtual environments, OpenVINO IR, GGUF and the llama.cpp build, attempts to replace accidentally installed GPU Torch with the CPU wheel, removes orphan experimental packages, installs and health-checks `v20-main.service`, and removes discarded unit files from `/etc/systemd/system`.

Success marker:

```text
V20 FINALIZED profile=main freed_bytes=<bytes> health=OK ...
```

RAG data, user dictionaries, benchmark results, the legacy service and the main venv are preserved. To skip shared-venv cleanup, run with `V20_PURGE_SHARED_EXPERIMENT_DEPS=false`.
