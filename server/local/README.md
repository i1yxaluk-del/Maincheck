# AI LibreOffice Suggester — local v2.4

This server is a **local Russian proofreader that applies minimal edits**. It is not a generic text-improvement service.

## Runtime model

Production uses exactly one systemd service and one Uvicorn process:

```text
.env
  ↓
ai-suggester.service
  ↓
uvicorn decision_app:app --host 0.0.0.0 --port 8000
```

No second inference server is used.

## Three supported stacks

### A — production

```text
raw selected text
  ↓
deterministic morphology candidates
  ↓
local BM25 + char-trigram retrieval of similar Russian GEC examples
  ↓
T-lite structured local-edit proposal
  ↓
merge candidates
  ↓
DecisionEngine
  ↓
minimal exact edits
```

Model: `t-tech/T-lite-it-2.1:q4_K_M` via Ollama.

The LLM is never asked to rewrite the paragraph. Retrieval is local and deterministic; it uses the repository's Russian GEC example bank with hashing embeddings plus BM25 word/trigram fusion, so no embedding model or external service is needed.

### G — experimental high-precision editor

```text
MorphDetector + narrow morphology rescue
  ↓
T-lite verifier
  ↓
DecisionEngine
```

G does not ask the LLM to discover new text changes. The LLM only votes on candidates already produced by deterministic logic. This is the conservative experiment for measuring precision-first correction.

### F — experimental surface corrector

```text
Spell-Corrector-RU-4B
  ↓
bounded local diff
  ↓
paragraph / line-break guard
  ↓
pymorphy3 morphology gate
  ↓
DecisionEngine
```

F remains isolated because it is a full-text generator and therefore has higher compute cost and a larger risk of changing valid word forms.

## Why this architecture

2025 Russian GEC research reports strong results from edit-based sequence tagging and shows that selecting similar correction examples with a GECToR-style retriever improves few-shot LLM correction. The LORuGEC paper reports up to 83% F0.5 for its best 5-shot setup and specifically reports gains from GECToR-based example selection. urlBEA 2025 paperhttps://aclanthology.org/2025.bea-1.38/

The 2025 Russian sequence-tagging work also reports state-of-the-art results on RU-Lang8 and GERA for its edit-based architecture. urlRussian sequence tagging paperhttps://aclanthology.org/2025.acl-srw.82/

BEA 2026 shows why a single aggregate score is insufficient: synthetic fine-tuning can raise overall F0.5 while sharply degrading individual grammar rules. Our service therefore keeps rule-level regression cases and destructive-edit tests in the repository. urlBEA 2026 diagnostichttps://synterr-nlp.github.io/papers/bea-2026/

For our hardware, the practical conclusion is to spend the expensive T-lite generation budget once, on a small edit-oriented prompt, and to move easy high-confidence work into deterministic local components.

## Installation and operation

Normal operation is only:

```bash
cd /home/service/llama/server/local
sudo systemctl restart ai-suggester.service
journalctl -u ai-suggester.service -n 120 --no-pager
```

After dependency/model changes, run the one-time installer:

```bash
cd /home/service/llama/server/local
bash install_experimental_models.sh
```

Select the stack only by editing `.env`:

```text
LLM_PRESET=A
```

or `F` / `G`, then restart the same service.

There is no preset-switching shell command in the production workflow. `/metrics` is diagnostic only and is not part of startup.

## Main settings

```text
LLM_PRESET=A
OLLAMA_URL=http://localhost:11434
NUM_THREADS=28
OLLAMA_NUM_CTX=2048
OLLAMA_NUM_PREDICT=192
OLLAMA_TIMEOUT=120
OLLAMA_WARMUP=true
OLLAMA_KEEP_ALIVE=24h
OLLAMA_TEMPERATURE=0

DECISION_MIN_CONFIDENCE=0.60
DECISION_MAX_CHANGES=12
DECISION_MAX_BEFORE_CHARS=120
MORPH_DETECTOR_ENABLED=true
USER_DICT_ENABLED=true
AUDIT_ENABLED=true
```

## Critical regression targets

The production corpus must detect:

```text
должностного лиц → должностного лица
```

and must never introduce the previously observed F transformations:

```text
изучена → изучено
dолжностного → должностных
деятельностей → деятельности
```

