# v4 correction architecture

The correction service is a cascade of independent candidate generators rather than one chat model:

1. A dedicated Russian spelling/punctuation model proposes a corrected draft.
2. That draft is converted to bounded local edits.
3. The Ollama language model proposes a second full-text correction for grammar, syntax and style.
4. The same model can also emit structured diagnostic edits for cases where a full draft hides the change.
5. A cross-model judge evaluates only model-generated edits; deterministic/high-confidence and specialist edits are kept independent.
6. The DecisionEngine is the final bounded applicator.

This mirrors current GEC practice: dedicated edit generators are materially better suited to proofreading than asking a general chat model to discover every error, while rule-level evaluation is required to prevent aggregate metrics from hiding blind spots.
