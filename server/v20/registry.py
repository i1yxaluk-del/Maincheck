from __future__ import annotations
from dataclasses import dataclass
import argparse

@dataclass(frozen=True)
class Profile:
    name: str
    entrypoint: str
    backend: str
    role: str

PROFILES = {
    "main": Profile("main", "v20.main_app:app", "hybrid-syntax-rag", "recommended"),
    "openvino": Profile("openvino", "v20.openvino_app:app", "openvino-seq2seq", "experimental"),
    "llama-json": Profile("llama-json", "v20.llama_json_app:app", "llama.cpp-json-edits", "experimental"),
}

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--check",action="store_true"); args=p.parse_args()
    if args.check:
        assert set(PROFILES)=={"main","openvino","llama-json"}
        assert len({x.entrypoint for x in PROFILES.values()})==3
        print("V20 REGISTRY OK profiles=3")
    else:
        for x in PROFILES.values(): print(f"{x.name}\t{x.backend}\t{x.role}\t{x.entrypoint}")
    return 0
if __name__=="__main__": raise SystemExit(main())
