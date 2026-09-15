# Maincheck v20 engine lab

Three mutually exclusive service profiles implement the existing `/suggest` multipart protocol.

## Profiles

- `main` — recommended hybrid-syntax-RAG route. It reuses the protected v12 edit pipeline, runs deterministic and specialist stages first, and calls bounded reasoning only for uncovered text.
- `openvino` — experimental encoder/decoder route. It exports `V20_OPENVINO_MODEL` to OpenVINO and accepts only bounded one-token replacements. This tests Intel CPU inference without Ollama.
- `llama-json` — experimental direct llama.cpp route. A dedicated `llama-server` returns position-bound JSON edits instead of a rewritten document.

Only one profile owns port 8000. The installer stops old and competing units first, rewrites `.env.v20` atomically, installs profile dependencies and starts through a transient background systemd unit.

## Install

```bash
sudo /home/service/llama/scripts/v20/install.sh main
journalctl -fu ai-suggester-v20-install-main
```

Success marker:

```text
V20 READY profile=main service=v20-main.service
```

OpenVINO:

```bash
sudo V20_OPENVINO_MODEL=ai-forever/sage-fredt5-large scripts/v20/install.sh openvino
```

llama.cpp requires a local GGUF path:

```bash
sudo V20_LLAMACPP_MODEL=/home/service/llama/models/v20/model.gguf scripts/v20/install.sh llama-json
```

## Benchmark

```bash
sudo /home/service/llama/scripts/v20/benchmark.sh main openvino llama-json
```

Results are written under `/home/service/llama/results/v20-<timestamp>/` as per-profile JSON and `summary.tsv` with pass rate, median and p95 latency.

## rollback

```bash
sudo systemctl disable --now v20-main.service v20-openvino.service v20-llama-json.service v20-llama-backend.service
sudo systemctl enable --now ai-suggester.service
```

The installer never deletes the old unit, RAG state, user dictionary or existing `.env`.
