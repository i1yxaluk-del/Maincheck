# v5 proofreading architecture

## Design goal

The service must maximize **error recall** without turning every request into a 4B multi-stage chat workflow. The previous v3.x path was recall-starved because the LLM had to enumerate exact edits. The previous X/Y path was operationally unusable because a 4B Transformers model was CPU/disk-offloaded.

## Model roles

1. `ai-forever/sage-fredt5-distilled-95m` is the fast spelling/punctuation/case specialist. It is a candidate generator, not a blind rewriter.
2. `Qwen/Qwen3.5-0.8B + synterr-nlp/bea2026-gec-adapters:v4_qwen35_08b_lorugec` is the measured Russian grammar/syntax specialist. The published BEA 2026 result is LORuGEC M2 F0.5=54.0 for this 0.8B adapter after SyntErr→LORuGEC continuation.
3. Existing Ollama T-lite or GigaChat is a third full-draft candidate source. It is not asked for JSON edit enumeration.

## Runtime path

```text
LibreOffice
   |
   v
FastAPI /suggest
   |
   +--> Local rules (high precision)
   +--> SAGE 95M --------------------+
   +--> SyntErr Qwen3.5-0.8B --------|--> bounded diff --> ranked candidates
   +--> optional Ollama full draft ---+
                                             |
                                             v
                                       DecisionEngine
                                             |
                                             v
                                      corrected text
```

A/B/X/Y differ only by which full-draft channel is enabled. X is the fast path; A/B add one larger local Ollama draft; Y adds both.

## Why no always-on judge

The v3.x judge became part of the recall bottleneck: a model-generated candidate could be rejected after it had already been found. v5 therefore uses deterministic source/category ranking. A future conflict-only judge can be added without making it mandatory for every request.

## Retrieval

The existing GEC example bank remains available to Ollama draft generation. It is a quality aid, not a correctness gate. This follows the LORuGEC finding that targeted corrected-example retrieval can materially improve Russian GEC.

## Operational target

The fast path should be dominated by the 0.8B GEC model and SAGE, not by a multi-minute 4B Transformers load. A/B/X/Y all continue to use the single existing systemd + Uvicorn + Ollama deployment model; no second web server is introduced.
