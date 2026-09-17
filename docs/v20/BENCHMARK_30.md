# Current-model 30-case quality benchmark

This corpus is a diagnostic baseline, not a set of rules the service is assumed to pass. It contains exactly 30 cases: 22 erroneous texts and 8 clean controls.

Sources:

- 13+ cases are failures or safety requirements supplied during target-host testing;
- normative government examples (`согласно приказу`, `по окончании`) are based on Gramota.ru guidance;
- legal-document categories are informed by the Ministry of Justice legal-technique material available through ConsultantPlus;
- remaining cases are synthetic minimal pairs designed to measure false positives.

Coverage: spelling, punctuation, agreement, government, official/legal style, terminology, dates, abbreviations, enumerations and line-break preservation.

Run against the currently active service without installing or switching profiles:

```bash
cd /home/service/llama
git pull --ff-only origin feat/v20-gec-engine-lab
bash scripts/v20/benchmark-current.sh
```

The script requires `stack=Z` by default and writes a detailed JSON report under `results/current-30-<timestamp>.json`. It prints total pass rate, positive recall, clean preservation, false-positive rate, layout preservation, explanation quality, latency, category breakdown and failed case IDs.

For a configured network/cloud endpoint, keep the same multipart contract and disable the local stack assertion:

```bash
V20_URL=https://server.example BENCHMARK_REQUIRE_STACK_Z=false bash scripts/v20/benchmark-current.sh
```
