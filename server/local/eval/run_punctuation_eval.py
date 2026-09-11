"""Evaluate the local RuPunct stage on error and clean official prose."""
from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path

from decision_engine import DecisionEngine
from rupunct_stage import RuPunctStage

CASES = Path(__file__).with_name("punctuation_cases.jsonl")

async def run():
    stage = RuPunctStage()
    outcomes = []
    for line in CASES.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        candidates = await stage.candidates(case["text"])
        result, _ = DecisionEngine(min_confidence=0.55).apply(case["text"], candidates)
        outcomes.append((case, result))
    return outcomes

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-error-recall", type=float, default=0.50)
    parser.add_argument("--max-clean-damaged", type=int, default=1)
    args = parser.parse_args()
    outcomes = asyncio.run(run())
    errors = [(c, r) for c, r in outcomes if c["text"] != c["expected"]]
    clean = [(c, r) for c, r in outcomes if c["text"] == c["expected"]]
    fixed = sum(r == c["expected"] for c, r in errors)
    damaged = sum(r != c["expected"] for c, r in clean)
    recall = fixed / len(errors) if errors else 0.0
    print(json.dumps({"error_total": len(errors), "error_exact": fixed,
                      "error_recall": round(recall, 4),
                      "clean_total": len(clean), "clean_damaged": damaged},
                     ensure_ascii=False, indent=2))
    return 0 if recall >= args.min_error_recall and damaged <= args.max_clean_damaged else 1

if __name__ == "__main__":
    raise SystemExit(main())
