import json
from pathlib import Path


def test_punctuation_corpus_has_errors_and_clean_controls():
    path = Path(__file__).parent / "eval" / "punctuation_cases.jsonl"
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(cases) >= 20
    assert sum(c["text"] != c["expected"] for c in cases) >= 10
    assert sum(c["text"] == c["expected"] for c in cases) >= 8
    assert len({c["id"] for c in cases}) == len(cases)
