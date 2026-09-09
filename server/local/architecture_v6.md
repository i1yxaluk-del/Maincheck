# v6 adaptive local-first proofreading architecture

The production failure mode observed in v5 is now explicit: the Qwen3.5/SyntErr adapter did not load cleanly on the installed Transformers stack, and generic full-text drafts can reorder/paraphrase an official sentence. A candidate generator must therefore be both measurable and bounded.

## Runtime

```text
LibreOffice -> FastAPI
                 |
                 +-> deterministic morphology/syntax rules
                 +-> SAGE 95M: spelling/punctuation candidate source
                 +-> ruBert-base MLM: masked-token candidate source
                 |
                 +-> if local recall is insufficient:
                       T-lite or GigaChat rescue draft
                       -> strict anti-rewrite diff
                 |
                 +-> ranked candidates -> DecisionEngine
```

The expensive Ollama stage is adaptive rather than mandatory. A/B use one rescue model; X is local-only; Y can use both rescue models. The Qwen3.5/SyntErr adapter remains disabled until its exact base/adapter compatibility is verified against the installed Transformers/PEFT stack.

## Why ruBert MLM

The Russian ruBert-base model is a 178M-parameter Russian masked-language encoder. It predicts alternatives at an existing word position instead of generating a new sentence. That makes it useful for detecting agreement, government, inflection and simple spelling substitutions while preventing sentence reordering by construction.

The model is a candidate generator only. Candidates are restricted by Russian morphology or small edit distance and pass through the existing DecisionEngine.

## Safety

Generic Ollama drafts are rescue candidates, not authoritative rewrites. `safe_diff` now rejects word reordering, multi-word paraphrases and broad substitutions even when character length stays similar.

## Operational target

The normal path should finish with local models. Ollama is invoked only when local evidence is weak. This avoids the previous 2–5 minute per-paragraph behavior and keeps one FastAPI/Uvicorn service.
