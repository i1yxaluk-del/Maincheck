# AI LibreOffice Suggester — local v2

Production uses **one** systemd service and **one** Uvicorn process:

```text
uvicorn decision_app:app --host 0.0.0.0 --port 8000
```

There is no second inference server and no second Uvicorn launcher.

## Three supported stacks

### A — production

```text
LibreOffice
  ↓ HTTP /suggest
FastAPI decision_app
  ↓
T-lite-it-2.1 via Ollama
  ↓ structured JSON edits only
DecisionEngine
  ↓ exact occurrence / confidence / overlap / protected-word gates
LibreOffice extension
```

Model: `t-tech/T-lite-it-2.1:q4_K_M`.

A is the only production stack. The LLM is never asked to return a rewritten paragraph; it returns local `before → after` edits. This makes the server apply only exact, bounded changes.

### F — experimental surface corrector

Model: `melsmm/Spell-Corrector-RU-4B`.

The model card describes the model as a Russian spelling/punctuation/case corrector and publishes the prompt `Исходный текст: ... Отредактируй исходный текст, исправив ошибки.` with low-temperature sampling. It is a full-text generator, so our adapter is deliberately stricter than the model itself: paragraph structure cannot change and any alphabetic token change must preserve the pymorphy3 morphology signature. See the model card: https://huggingface.co/melsmm/Spell-Corrector-RU-4B

F is an experiment, not production.

### G — experimental conservative detector

```text
raw text
  ↓
MorphDetector / local edit candidates
  ↓
T-lite verifier
  ↓
DecisionEngine
  ↓
minimal changes
```

T-lite is only a verifier. It never rewrites the whole paragraph. G is expected to have lower recall than A but a lower false-positive risk.

## Why D/E/C were removed

D (Qwen3.5-4B + SyntErr→LORuGEC LoRA) is academically interesting: the published BEA 2026 adapter card reports 75.3 M2 F0.5 on LORuGEC test for Qwen3.5-4B with SyntErr→LORuGEC. The same card shows the official PEFT loading pattern. https://huggingface.co/synterr-nlp/bea2026-gec-adapters

It is not part of the supported local stack because the current production host produced missing LoRA adapter keys and a real request took about 262 seconds in the observed run. A model that loads incorrectly or exceeds LibreOffice's practical response budget must not be a production dependency.

E was a local MorphDetector wrapper rather than the published RussianGEC sequence-tagger checkpoint. The public RussianGEC_SeqTagger repository contains training/inference code, but not a ready checkpoint suitable for this deployment. https://github.com/ReginaNasyrova/RussianGEC_SeqTagger

C added another Ollama generation hop. On this host the additional model did not justify the extra latency and complexity compared with keeping one high-quality generator plus deterministic gates.

## Install

```bash
cd /home/service/llama/server/local
source venv/bin/activate
pip install -r requirements.txt
```

For F, cache the Hugging Face model once:

```bash
bash install_experimental_models.sh
```

The installer also pins `setuptools==81.0.0` because the current Natasha runtime still imports `pkg_resources`.

## Switch stacks

```bash
./scripts/switch_llm_preset.sh A
./scripts/switch_llm_preset.sh G
./scripts/switch_llm_preset.sh F
sudo systemctl restart ai-suggester.service
```

Check:

```bash
curl -s http://localhost:8000/metrics | python3 -m json.tool
journalctl -u ai-suggester.service -n 120 --no-pager
```

## Main runtime settings

```text
LLM_PRESET=A
OLLAMA_URL=http://localhost:11434
NUM_THREADS=28
OLLAMA_NUM_CTX=2048
OLLAMA_NUM_PREDICT=384
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

## Regression cases

The test suite explicitly guards against the previously observed destructive F output:

```text
изучена         → изучено
должностного    → должностных
деятельностей   → деятельности
```

Those changes must never reach `DecisionEngine` from F.

Run locally:

```bash
pytest -q server/local/test_experimental_backend.py
```

And the stack smoke preflight:

```bash
bash server/local/install_experimental_models.sh
```
