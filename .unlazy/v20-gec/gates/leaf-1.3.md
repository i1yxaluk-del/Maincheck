# Gates: v20 benchmark

OWNS: tests/v20/**, scripts/v20/benchmark.sh

Scope: compare all profiles against regression and clean-text fixtures with latency statistics

- [ ] G1: fixture corpus is valid and includes positive and negative controls
  CHECK: PYTHONPATH=server python3 -m v20.benchmark --validate-corpus tests/v20/cases.jsonl
  EXPECT: V20 CORPUS OK
  EVIDENCE: pending
- [ ] G2: benchmark shell passes syntax validation
  CHECK: bash -n scripts/v20/benchmark.sh
  EXPECT: 
  EVIDENCE: pending
