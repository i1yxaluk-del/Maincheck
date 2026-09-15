# Gates: v20 documentation and CI

OWNS: docs/v20/**, .github/workflows/v20-engine-lab.yml

Scope: document architecture, deployment, rollback, metrics and execute static checks in CI

- [ ] G1: architecture documents all three profiles and rollback
  CHECK: python3 scripts/v20/verify_static.py --docs
  EXPECT: V20 DOCS OK
  EVIDENCE: pending
- [ ] G2: CI workflow names root verification commands
  CHECK: python3 scripts/v20/verify_static.py --ci
  EXPECT: V20 CI OK
  EVIDENCE: pending
