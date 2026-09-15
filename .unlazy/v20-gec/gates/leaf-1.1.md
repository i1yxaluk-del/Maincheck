# Gates: v20 engine profiles

OWNS: server/v20/**

Scope: implement one recommended and two structurally different correction engines behind one API contract

- [ ] G1: registry exposes exactly main, openvino and llama-json profiles
  CHECK: PYTHONPATH=server python3 -m v20.registry --check
  EXPECT: V20 REGISTRY OK profiles=3
  EVIDENCE: pending
- [ ] G2: engine modules compile
  CHECK: python3 -m compileall -q server/v20
  EXPECT: 
  EVIDENCE: pending
