from __future__ import annotations

import logging

from decision_engine import EditCandidate

log = logging.getLogger("ai_suggester.syntax_candidates")

UD_TO_PYMORPHY = {
    "Nom": "nomn", "Gen": "gent", "Dat": "datv", "Acc": "accs",
    "Ins": "ablt", "Loc": "loct", "Par": "loct", "Voc": "voct",
    "Sing": "sing", "Plur": "plur",
    "Masc": "masc", "Fem": "femn", "Neut": "neut",
}


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
        if not _is_noun(head.pos) or doc.is_clearly_non_attributive(i, head_idx):
            continue

        src = [p for p in morph.parse(tok.text) if p.is_known]
        if not src:
            continue
        feats = tok.feats_dict
        head_feats = head.feats_dict
        src_case = feats.get("Case")
        src_number = feats.get("Number")
        src_gender = feats.get("Gender")
        target_case = head_feats.get("Case") or src_case
        target_number = head_feats.get("Number") or src_number
        target_gender = head_feats.get("Gender") or src_gender

        mismatch = (
            bool(src_case and target_case and src_case != target_case)
            or bool(src_number and target_number and src_number != target_number)
            or bool(src_gender and target_gender and src_gender != target_gender)
        )
        if not mismatch:
            continue

        grammemes = {
            UD_TO_PYMORPHY[value]
            for value in (target_case, target_number, target_gender)
            if value in UD_TO_PYMORPHY
        }
        if not grammemes:
            continue
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
            f"dependency {tok.rel} + согласование с `{head.text}`",
        ))
    return out
