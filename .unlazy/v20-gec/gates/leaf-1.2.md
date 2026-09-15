# Gates: v20 deployment

OWNS: server/local/v20-*.service, scripts/v20/install.sh, scripts/v20/install-worker.sh, scripts/v20/verify_static.py

Scope: install profiles safely in the background and expose readiness in logs

- [ ] G1: all service units name valid v20 entrypoints
  CHECK: python3 scripts/v20/verify_static.py --services
  EXPECT: V20 SERVICES OK
  EVIDENCE: pending
- [ ] G2: installer and worker pass shell syntax validation
  CHECK: bash -n scripts/v20/install.sh && bash -n scripts/v20/install-worker.sh
  EXPECT: 
  EVIDENCE: pending
- [ ] G3: sudo-bearing deployment steps receive administrator review
  EVIDENCE: pending
