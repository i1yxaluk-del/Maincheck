# Plan: v20 Russian GEC engine lab

Scope: v20-gec
Depth: tree 2
Mode: orchestrated (sequential fallback; host has no native subagent launch surface)

## Contract
- Interfaces: every engine exposes POST /suggest and GET /health; benchmark consumes the existing corrected-text protocol.
- Ownership: disjoint leaf paths below.
- Dependencies: services depend on engine entrypoints; benchmark integration depends on all profiles.
- Host launch mode: sequential fallback.
- Toolchain: Python 3.10+, bash, systemd, curl; repository root as CWD.
- Conventions: no secrets in env templates; profiles main, openvino, llama-json; safe local edits only.
- Manual review: deployment scripts require administrator review before sudo execution.

## Current contract inventory
Contract revision: 1.

| ID | Required outcome or constraint | Owner | Observing gate or manual review | Disposition | Revision |
|---|---|---|---|---|---|
| C1 | One recommended production profile and two structurally different experimental profiles | leaf-1.1 | leaf-1.1:G1 | ACTIVE | 1 |
| C2 | Separate entrypoints and service units for all profiles | leaf-1.2 | leaf-1.2:G1 | ACTIVE | 1 |
| C3 | Background installer rewrites env, installs dependencies, frees resources, enables selected services and logs readiness | leaf-1.2 | leaf-1.2:G2 | ACTIVE | 1 |
| C4 | Test script contains reported and representative Russian GEC cases | leaf-1.3 | leaf-1.3:G1 | ACTIVE | 1 |
| C5 | Benchmark reports quality, false positives and latency for profile comparison | leaf-1.3 | leaf-1.3:G2 | ACTIVE | 1 |
| C6 | Architecture, operations and limitations are documented | leaf-1.4 | leaf-1.4:G1 | ACTIVE | 1 |
| C7 | Existing local and network routes remain available | node-1 | node-1:N2 | ACTIVE | 1 |

## Tree
- 1 v20 engine lab .............. GATES.md
  - 1.1 engine profiles ......... gates/leaf-1.1.md
  - 1.2 deployment .............. gates/leaf-1.2.md
  - 1.3 benchmark ............... gates/leaf-1.3.md
  - 1.4 documentation and CI .... gates/leaf-1.4.md

## Leaf dispatch table
| Leaf | Owns | Needs | Tier | Planned wave | State |
|---|---|---|---|---|---|
| 1.1 | server/v20/** | - | judgment | 1 | READY |
| 1.2 | server/local/v20-*.service, scripts/v20/** | 1.1 | judgment | 2 | WAITING |
| 1.3 | tests/v20/**, scripts/v20/benchmark.sh | 1.1 | judgment | 2 | WAITING |
| 1.4 | docs/v20/**, .github/workflows/v20-engine-lab.yml | 1.1, 1.2, 1.3 | judgment | 3 | WAITING |
