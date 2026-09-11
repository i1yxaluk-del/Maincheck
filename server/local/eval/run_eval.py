"""Оффлайн-оценка качества локального корректора (v9).

Зачем
=====
До v9 в репозитории не было метрики качества: каждая версия стека
(v5…v8) оценивалась на одном-двух предложениях из отчёта, а регрессии на
корректном тексте вообще не измерялись. Именно поэтому правило
согласования с нулевой точностью прожило несколько релизов.

Что считаем
===========
* **FP-rate на корректном тексте** — доля заведомо правильных предложений,
  которые система изменила. Для корректора документов это главная
  метрика: любая правка корректного текста дороже пропущенной ошибки,
  потому что подрывает доверие к инструменту.
* **Exact match** на предложениях с ошибками.
* **Token-level P / R / F0.5** по множествам правок (упрощённый ERRANT).
  F0.5, а не F1 — в GEC точность весит вдвое больше полноты.

Запуск
======
    PYTHONPATH=../.. python -m eval.run_eval                # только детерминированные стадии
    PYTHONPATH=../.. python -m eval.run_eval --stack full   # + SAGE/GEC/Ollama (нужен сервер)
    PYTHONPATH=../.. python -m eval.run_eval --max-fp 0     # CI-гейт

Оффлайн-режим не требует ни сети, ни Ollama, ни transformers — только
pymorphy3. Поэтому его можно и нужно держать в CI.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

CASES = Path(__file__).resolve().parent / "cases.jsonl"
WORD_SPLIT = __import__("re").compile(r"\w+|[^\w\s]", __import__("re").UNICODE)


@dataclass
class Outcome:
    case_id: str
    kind: str
    text: str
    expected: str
    produced: str
    accepted: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.produced != self.text

    @property
    def exact(self) -> bool:
        return self.produced == self.expected


def load_cases(path: Path = CASES) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def edit_set(source: str, target: str) -> set[tuple[int, str, str]]:
    """Множество правок как (позиция в словах источника, before, after)."""
    a = WORD_SPLIT.findall(source)
    b = WORD_SPLIT.findall(target)
    out: set[tuple[int, str, str]] = set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        out.add((i1, " ".join(a[i1:i2]), " ".join(b[j1:j2])))
    return out


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision <= 0 and recall <= 0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


async def run_deterministic(cases: list[dict], min_confidence: float, max_changes: int) -> list[Outcome]:
    from decision_engine import DecisionEngine
    from local_rules import LocalRuleEngine
    from spellcheck import DictionarySpellChecker
    from verification import GenerativeGuard

    rules = LocalRuleEngine()
    speller = DictionarySpellChecker()
    guard = GenerativeGuard()
    outcomes: list[Outcome] = []
    for case in cases:
        text = case["text"]
        candidates = rules.candidates(text) + speller.candidates(text)
        engine = DecisionEngine(
            min_confidence=min_confidence, max_changes=max_changes, guard=guard,
        )
        produced, accepted = engine.apply(text, candidates)
        outcomes.append(Outcome(
            case_id=case["id"], kind=case["kind"], text=text,
            expected=case.get("expected", text), produced=produced,
            accepted=[(c.before, c.after, c.category) for c in accepted],
        ))
    return outcomes


async def run_full(cases: list[dict], min_confidence: float, max_changes: int) -> list[Outcome]:
    from decision_engine import DecisionEngine
    from hybrid_editor import HybridRouter
    from verification import GenerativeGuard
    import os

    router = HybridRouter(os.getenv("LLM_PRESET", "A"))
    await router.warmup()
    guard = GenerativeGuard()
    outcomes: list[Outcome] = []
    for case in cases:
        text = case["text"]
        candidates = await router.candidates(text, "")
        engine = DecisionEngine(
            min_confidence=min_confidence, max_changes=max_changes, guard=guard,
        )
        produced, accepted = engine.apply(text, candidates)
        outcomes.append(Outcome(
            case_id=case["id"], kind=case["kind"], text=text,
            expected=case.get("expected", text), produced=produced,
            accepted=[(c.before, c.after, c.category) for c in accepted],
        ))
    return outcomes


def report(outcomes: list[Outcome], verbose: bool) -> dict:
    clean = [o for o in outcomes if o.kind == "clean"]
    errors = [o for o in outcomes if o.kind == "error"]

    damaged = [o for o in clean if o.changed]
    tp = fp = fn = 0
    for o in errors:
        gold = edit_set(o.text, o.expected)
        system = edit_set(o.text, o.produced)
        tp += len(gold & system)
        fp += len(system - gold)
        fn += len(gold - system)
    for o in clean:
        system = edit_set(o.text, o.produced)
        fp += len(system)

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    stats = {
        "clean_total": len(clean),
        "clean_damaged": len(damaged),
        "clean_fp_rate": round(len(damaged) / len(clean), 4) if clean else 0.0,
        "error_total": len(errors),
        "error_exact": sum(1 for o in errors if o.exact),
        "edits_tp": tp,
        "edits_fp": fp,
        "edits_fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f05": round(f_beta(precision, recall, 0.5), 4),
    }

    print("=" * 78)
    print("Оценка локального корректора")
    print("=" * 78)
    for key, value in stats.items():
        print(f"  {key:<16} {value}")

    if damaged:
        print("\n--- ИСПОРЧЕННЫЙ КОРРЕКТНЫЙ ТЕКСТ (должно быть пусто) ---")
        for o in damaged:
            print(f"  [{o.case_id}] {o.text}")
            print(f"      -> {o.produced}")
            for before, after, category in o.accepted:
                print(f"         {before!r} -> {after!r}  ({category})")

    if verbose:
        print("\n--- ПРЕДЛОЖЕНИЯ С ОШИБКАМИ ---")
        for o in errors:
            mark = "OK  " if o.exact else ("part" if o.changed else "miss")
            print(f"  [{mark}] {o.case_id}")
            if not o.exact:
                print(f"      ожидалось: {o.expected}")
                print(f"      получено : {o.produced}")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", choices=["deterministic", "full"], default="deterministic")
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--max-changes", type=int, default=12)
    parser.add_argument("--max-fp", type=int, default=None,
                        help="CI-гейт: максимум испорченных корректных предложений")
    parser.add_argument("--min-exact", type=int, default=None,
                        help="CI-гейт: минимум точных исправлений на ошибочных примерах")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true", help="печатать только JSON-сводку")
    args = parser.parse_args()

    cases = load_cases()
    runner = run_deterministic if args.stack == "deterministic" else run_full
    outcomes = asyncio.run(runner(cases, args.min_confidence, args.max_changes))

    if args.json:
        stats = report(outcomes, verbose=False)
        print(json.dumps(stats, ensure_ascii=False))
    else:
        stats = report(outcomes, verbose=args.verbose)

    failed = False
    if args.max_fp is not None and stats["clean_damaged"] > args.max_fp:
        print(f"\nFAIL: испорчено корректных предложений {stats['clean_damaged']} > {args.max_fp}")
        failed = True
    if args.min_exact is not None and stats["error_exact"] < args.min_exact:
        print(f"\nFAIL: точных исправлений {stats['error_exact']} < {args.min_exact}")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
