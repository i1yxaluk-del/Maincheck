# Gates: v20 Russian GEC engine lab

Scope: integrate three engines, deployment automation and comparative benchmark

- [ ] R1: all Python and shell artifacts pass static validation
  CHECK: python3 scripts/v20/verify_static.py
  EXPECT: V20 STATIC OK
  EVIDENCE: pending
- [ ] R2: benchmark self-test validates metrics and failure handling
  CHECK: PYTHONPATH=server python3 -m v20.benchmark --self-test
  EXPECT: V20 BENCHMARK SELFTEST OK
  EVIDENCE: pending
- [ ] R3: deployment commands and sudo effects receive administrator review
  EVIDENCE: pending
