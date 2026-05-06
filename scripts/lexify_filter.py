"""v1.7.2: фильтр alexanderpl/Lexify_RuGEC под админ/деловой стиль с
балансировкой по типам морфо-агрементов.

Шаги:
  1. Загружаем 593 550 пар.
  2. Фильтр сорсов: оставляем только src=1 (academic/technical) и
     src=19 (narrative). Остальное — L2 / colloquial / single-word.
  3. На каждой паре считаем word-level diff. Если diff содержит хотя
     бы один morph-агремент (same lemma, разный case ИЛИ разный
     number) — пара полезна нам.
  4. Дополнительно: классифицируем все diff-пары и считаем категории.
     В выдачу берём СБАЛАНСИРОВАННОЕ количество каждого типа, не
     просто top-K.

Output:
  data/lexify_admin.jsonl — итоговый банк в нашем GecPair-формате.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

try:
    import pymorphy3
    morph = pymorphy3.MorphAnalyzer()
except ImportError:
    print("ERROR: pymorphy3 not installed", file=sys.stderr)
    sys.exit(1)


# Сорсы, которые признаём пригодными для админ-стиля (на основе
# ручной выборки в этом сеансе).
GOOD_SOURCES = {1, 19}

# L2-маркеры (если входной текст содержит — почти наверняка L2)
L2_MARKERS = re.compile(
    r"\b(семья|обучение|друзей|русский язык|хочу|кофе|школьник|обычно|"
    r"кафе|отдыхать|каникул|хобби)\b",
    re.IGNORECASE,
)

# Минимальная длина входа в символах (короткие — single-word typos,
# нам бесполезны для retrieval'а)
MIN_INPUT_LEN = 30
MAX_INPUT_LEN = 400

# Количество правок-токенов: 1-3 ок, больше — паре сложно прайминг
MAX_DIFF_TOKENS = 4


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def classify_diff(b_word: str, a_word: str) -> str | None:
    """Классифицирует пару (before, after) по морф-категории.
    
    Возвращает один из:
      'number_agreement'  — одна лемма, разный number, тот же case
      'case_agreement'    — одна лемма, тот же number, разный case
      'verb_form'         — одна лемма, разный tense/number/person (для глаголов)
      'lexical'           — разные леммы (не наш кейс, скип)
      'eyo'               — только ё/е разница (не наш кейс)
      None                — не классифицируется (e.g., orthography fix)
    """
    if b_word.lower() == a_word.lower():
        return None
    if b_word.lower().replace("ё", "е") == a_word.lower().replace("ё", "е"):
        return "eyo"
    pb = morph.parse(b_word)
    pa = morph.parse(a_word)
    if not pb or not pa:
        return None
    bp = pb[0]
    ap = pa[0]
    if bp.normal_form != ap.normal_form:
        return "lexical"
    # одна лемма
    if bp.tag.POS != ap.tag.POS:
        return "pos_change"
    if bp.tag.POS in ("VERB", "INFN", "GRND", "PRTF", "PRTS"):
        return "verb_form"
    # noun/adj
    if bp.tag.case == ap.tag.case and bp.tag.number != ap.tag.number:
        return "number_agreement"
    if bp.tag.case != ap.tag.case and bp.tag.number == ap.tag.number:
        return "case_agreement"
    if bp.tag.case != ap.tag.case and bp.tag.number != ap.tag.number:
        return "case_and_number"
    return None


def diff_pairs(inp: str, out: str) -> list[tuple[str, str]]:
    """Простейший word-level diff — выравниваем по индексу. Если
    длины токенов различаются — пропускаем (insert/delete = не наш кейс)."""
    bt = tokenize(inp)
    at = tokenize(out)
    if len(bt) != len(at):
        return []
    pairs = []
    for b, a in zip(bt, at):
        if b != a:
            pairs.append((b, a))
    return pairs


def main():
    print("Loading alexanderpl/Lexify_RuGEC ...", file=sys.stderr)
    ds = load_dataset("alexanderpl/Lexify_RuGEC", split="train")
    print(f"Total pairs: {len(ds)}", file=sys.stderr)

    # 1. Фильтр сорсов
    by_source = [r for r in ds if r["source"] in GOOD_SOURCES]
    print(f"After source filter: {len(by_source)}", file=sys.stderr)

    # 2. Фильтр длины + L2-маркеры
    candidates = []
    for r in by_source:
        inp = r["input"].strip()
        out = r["output"].strip()
        if not inp or not out or inp == out:
            continue
        if not (MIN_INPUT_LEN <= len(inp) <= MAX_INPUT_LEN):
            continue
        if L2_MARKERS.search(inp):
            continue
        candidates.append((inp, out))
    print(f"After length/L2 filter: {len(candidates)}", file=sys.stderr)

    # 3. Diff classification
    by_category: dict[str, list[tuple[str, str]]] = defaultdict(list)
    skip = {"lexical", "eyo", "pos_change", None}
    for inp, out in candidates:
        pairs = diff_pairs(inp, out)
        if not pairs or len(pairs) > MAX_DIFF_TOKENS:
            continue
        cats = [classify_diff(b, a) for b, a in pairs]
        # Берём первую интересную категорию
        primary = next((c for c in cats if c not in skip), None)
        if primary is None:
            continue
        by_category[primary].append((inp, out))

    print("\nDiff category distribution:", file=sys.stderr)
    for cat, items in sorted(by_category.items(), key=lambda x: -len(x[1])):
        print(f"  {cat:20s}: {len(items)}", file=sys.stderr)

    # 4. Балансировка: до N пар на категорию
    PER_CAT = 1500
    final = []
    for cat, items in by_category.items():
        sample = items[:PER_CAT]
        for inp, out in sample:
            final.append({
                "wrong": inp,
                "right": out,
                "rule": f"Lexify-{cat}",
                "definition": f"Автоматически отобранная пара из Lexify_RuGEC, категория: {cat}",
                "section": "Grammar",
            })

    print(f"\nFinal pairs (after balance): {len(final)}", file=sys.stderr)

    out_path = Path("server/shared/gec_seed/lexify_admin.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for entry in final:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
