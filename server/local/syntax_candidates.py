from __future__ import annotations

import logging
from typing import Any

from decision_engine import EditCandidate

log = logging.getLogger("ai_suggester.syntax_candidates")


def _is_modifier(pos: str, rel: str) -> bool:
    return pos in {"ADJ", "PRON", "DET"} or rel in {"amod", "det"}


def _is_noun(pos: str) -> bool:
    return pos in {"NOUN", "PROPN"}


def candidates(text: str) -> list[EditCandidate]:
    """Find modifier/head agreement errors from a dependency tree.

    Unlike the old adjacent-token rescue this follows real dependency links,
    so phrases such as `профессиональной служебная ... подготовки` are seen
    even when the adjective and noun are separated by several words.
    """
    try:
        import pymorphy3
        from shared.syntax_parser import get_syntax_parser
        parser = get_syntax_parser()
        if not parser.available:
            return []
        doc = parser.parse(text)
        if doc is None:
            return []
        morph = pymorphy3.MorphAnalyzer()
    except Exception as exc:
        log.warning("dependency candidate detector unavailable: %s", exc)
        return []

    out: list[EditCandidate] = []
    for i, tok in enumerate(doc.tokens):
        if not _is_modifier(tok.pos, tok.rel):
            continue
        head_idx = tok.head_idx
        if head_idx < 0 or head_idx >= len(doc.tokens):
            continue
        head = doc.tokens[head_idx]
        if not _is_noun(head.pos):
            continue
        if doc.is_clearly_non_attributive(i, head_idx):
            continue

        src = morph.parse(tok.text)
        if not src:
            continue
        src = [p for p in src if p.is_known]
        if not src:
            continue
        p = src[0]
        feats = tok.feats_dict
        case = feats.get("Case")
        number = feats.get("Number")
        gender = feats.get("Gender")
        head_feats = head.feats_dict
        target_case = head_feats.get("Case") or case
        target_number = head_feats.get("Number") or number
        target_gender = head_feats.get("Gender") or gender

        mismatch = False
        if case and target_case and case != target_case:
            mismatch = True
        if number and target_number and number != target_number:
            mismatch = True
        if gender and target_gender and gender not in {"", target_gender}:
            mismatch = True
        if not mismatch:
            continue

        grammemes = set()
        mapping = {"case": target_case, "number": target_number, "gender": target_gender}
        for attr in mapping.values():
            if attr:
                # pymorphy3 grammeme names are lower-case.
                grammemes.add(str(attr).lower())
        fixed = None
        for parse in src[:3]:
            try:
                candidate = parse.inflect(grammemes)
            except Exception:
                candidate = None
            if candidate and candidate.word != tok.text:
                fixed = candidate.word
                break
        if not fixed or fixed.casefold() == tok.text.casefold():
            continue
        if tok.text[:1].isupper():
            fixed = fixed[:1].upper() + fixed[1:]
        out.append(EditCandidate(
            tok.text,
            fixed,
            0.985,
            "syntax-agreement",
            f"dependency {tok.rel} + морфологическое согласование с `{head.text}`",
        ))
    return out
