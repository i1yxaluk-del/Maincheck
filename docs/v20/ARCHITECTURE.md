# v20 architecture decision

## Goal

Compare architectures, not prompts, on the same Russian official-text corpus and existing Writer protocol.

## Main: hybrid-syntax-RAG

Low-cost deterministic morphology, LanguageTool, SAGE/RUPunct and normative RAG create local candidates. The 7B reasoner is coverage-gated and time-bounded. This profile has the strongest compatibility with current deployment and is the recommended migration baseline.

## Experiment A: OpenVINO seq2seq

A separate FastAPI process loads an encoder-decoder correction model through Optimum Intel/OpenVINO. It does not use Ollama. Full rewrites are not trusted: only unique bounded token replacements are projected back to the original text. This measures Intel-optimized inference and larger SAGE-family models while preserving layout safety.

## Experiment B: direct llama.cpp JSON edits

A dedicated llama-server owns the GGUF and exposes OpenAI-compatible chat. The interpreter requests offset-bound edits, validates every `before` slice and rejects overlaps, newlines and large replacements. This removes Ollama from the path and permits NUMA, quantization and speculative-decoding experiments.

## Resource isolation

Profiles are mutually exclusive. The background installer stops old v20 units and the legacy service before loading a new kernel. OpenVINO and llama.cpp profiles stop Ollama to recover RAM. The llama backend is bound to loopback.

## Selection rule

Do not select by one sentence. Use pass rate with clean negative controls, formatting preservation, median latency and p95. A candidate cannot become default if it improves recall by corrupting protected terms or clean text.

## Known limitations

The public RussianGEC sequence-tagger code does not ship production weights, so v20 does not pretend it is immediately deployable. The OpenVINO profile is a measurable bridge; a future trained ruRoberta tagger can implement the same API. Model downloads and llama.cpp source builds require network access during installation. Administrator review is required before executing sudo scripts.
