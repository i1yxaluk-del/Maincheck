from __future__ import annotations

import logging
import re
from typing import Any

from decision_engine import EditCandidate

log = logging.getLogger("ai_suggester.local_rules")
WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z]+(?:[-/][А-Яа-яЁёA-Za-z]+)*")
YEAR_RE = re.compile(r"\b(\d{4})\s+([А-Яа-яЁё-]+)\s+(год(?:а|у|ом|е|ов|ы)?|лет)\b")
PO_ONE_RE = re.compile(r"\bпо\s+одна\b", re.IGNORECASE)
REQUIRES_COMMA_RE = re.compile(r"\bтребует,\s+(?P<word>[А-Яа-яЁё]+)")


class LocalRuleEngine:
    """High-precision candidates for constructions where deterministic grammar wins.

    The rules are intentionally narrower than a full grammar checker. Their purpose is
    to guarantee recall for a few objective constructions that generic LLMs repeatedly
    missed in production, while keeping every edit local and auditable.
    """

    def __init__(self) -> None:
        self.morph = None
        try:
            import pymorphy3
            self.morph = pymorphy3.MorphAnalyzer()
        except Exception as exc:  # pragma: no cover
            log.warning("LocalRuleEngine: pymorphy3 unavailable: %s", exc)

    @property
    def available(self) -> bool:
        return self.morph is not None

    @staticmethod
    def _is_adj(parse: Any) -> bool:
        return any(x in str(parse.tag) for x in ("ADJF", "ADJS", "PRTF", "PRTS"))

    @staticmethod
    def _is_noun(parse: Any) -> bool:
        return "NOUN" in str(parse.tag)

    def _noun_parses(self, word: str) -> list[Any]:
        if not self.morph:
            return []
        return [p for p in self.morph.parse(word) if p.is_known and self._is_noun(p)]

    def _adj_parses(self, word: str) -> list[Any]:
        if not self.morph:
            return []
        return [p for p in self.morph.parse(word) if p.is_known and self._is_adj(p)]

    def _year_phrase(self, text: str) -> list[EditCandidate]:
        if not self.morph:
            return []
        out: list[EditCandidate] = []
        for match in YEAR_RE.finditer(text):
            year, adjective, noun = match.group(1), match.group(2), match.group(3)
            try:
                if not 1900 <= int(year) <= 2100:
                    continue
            except ValueError:
                continue
            adj = self._adj_parses(adjective)
            nouns = self._noun_parses(noun)
            if not adj or not nouns:
                continue
            year_noun = next((p for p in nouns if p.normal_form == "год"), None)
            if year_noun is None:
                continue
            # In Russian constructions like "2026 учебный год" the year
            # governs singular nominative adjective + noun.
            target = {"nomn", "sing"}
            if str(year_noun.tag.gender):
                target.add("masc")
            fixed_adj = None
            for p in adj[:5]:
                candidate = p.inflect(target)
                if candidate and candidate.word != adjective:
                    fixed_adj = candidate.word
                    break
            if fixed_adj:
                if adjective[:1].isupper():
                    fixed_adj = fixed_adj[:1].upper() + fixed_adj[1:]
                before = f"{year} {adjective} {noun}"
                after = f"{year} {fixed_adj} год"
                out.append(EditCandidate(before, after, 0.995, "rule-year", "годовая дата: числительное + нормативная форма 'учебный год'"))
        return out

    def _modifier_noun(self, text: str) -> list[EditCandidate]:
        """Adjacent and coordinated adjective+noun agreement with strong guards."""
        if not self.morph:
            return []
        matches = list(WORD_RE.finditer(text))
        out: list[EditCandidate] = []
        for noun_idx, noun_match in enumerate(matches):
            noun = noun_match.group(0)
            nouns = self._noun_parses(noun)
            if not nouns or noun_idx == 0:
                continue
            noun_parse = nouns[0]
            # Look backwards through a short coordinated modifier chain.
            j = noun_idx - 1
            modifier_indexes: list[int] = []
            steps = 0
            while j >= 0 and steps < 5:
                word = matches[j].group(0)
                if word.casefold() == "и":
                    j -= 1
                    steps += 1
                    continue
                parses = self._adj_parses(word)
                if not parses:
                    break
                modifier_indexes.append(j)
                j -= 1
                steps += 1
            if not modifier_indexes:
                continue
            # Skip multiword/prepositional structures unless at least two
            # modifiers in the same short chain agree with the head; this is
            # the high-precision signal that one modifier is actually wrong.
            if len(modifier_indexes) == 1:
                mod_word = matches[modifier_indexes[0]].group(0)
                gap = text[matches[modifier_indexes[0]].end():noun_match.start()]
                if any(ch in gap for ch in ",;:") or "-" in mod_word:
                    continue
            parses_by_idx = {idx: self._adj_parses(matches[idx].group(0)) for idx in modifier_indexes}
            compatible_any = []
            for idx in modifier_indexes:
                word = matches[idx].group(0)
                parses = parses_by_idx[idx]
                if any(
                    p.tag.case == noun_parse.tag.case
                    and p.tag.number == noun_parse.tag.number
                    and (not p.tag.gender or not noun_parse.tag.gender or p.tag.gender == noun_parse.tag.gender)
                    for p in parses
                ):
                    compatible_any.append(idx)
            # In a coordinated chain, use the noun as head and fix only
            # modifiers that have a unique obvious inflection to the head.
            if len(modifier_indexes) > 1 and not compatible_any:
                continue
            for idx in reversed(modifier_indexes):
                word = matches[idx].group(0)
                parses = parses_by_idx[idx]
                fixed = None
                for p in parses[:5]:
                    grammemes = set()
                    if noun_parse.tag.case:
                        grammemes.add(noun_parse.tag.case)
                    if noun_parse.tag.number:
                        grammemes.add(noun_parse.tag.number)
                    if noun_parse.tag.gender:
                        grammemes.add(noun_parse.tag.gender)
                    candidate = p.inflect(grammemes)
                    if candidate and candidate.word != word:
                        fixed = candidate.word
                        break
                if not fixed:
                    continue
                if word[:1].isupper():
                    fixed = fixed[:1].upper() + fixed[1:]
                out.append(EditCandidate(word, fixed, 0.985, "rule-agreement", f"согласование определения с существительным «{noun}»"))
        return out

    @staticmethod
    def _po_one(text: str) -> list[EditCandidate]:
        return [EditCandidate(m.group(0), m.group(0).replace("одна", "одной"), 0.995, "rule-government", "форма после предлога «по»") for m in PO_ONE_RE.finditer(text)]

    def _requires_comma(self, text: str) -> list[EditCandidate]:
        if not self.morph:
            return []
        out: list[EditCandidate] = []
        for m in REQUIRES_COMMA_RE.finditer(text):
            word = m.group("word")
            parses = self.morph.parse(word)
            # "требует" normally governs a direct genitive object here;
            # when a genitive noun follows immediately, an intervening comma
            # is a strong punctuation anomaly rather than a clause boundary.
            is_genitive_noun = any(p.is_known and self._is_noun(p) and p.tag.case == "gent" for p in parses)
            if is_genitive_noun:
                before = ", "
                start = m.start() + m.group(0).find(",")
                if start >= 0:
                    out.append(EditCandidate(before, " ", 0.97, "rule-punctuation", "запятая между сказуемым «требует» и его генитивным дополнением"))
        return out

    def candidates(self, text: str) -> list[EditCandidate]:
        out: list[EditCandidate] = []
        seen: set[tuple[str, str, str]] = set()
        for group in (self._year_phrase(text), self._modifier_noun(text), self._po_one(text), self._requires_comma(text)):
            for c in group:
                key = (c.before, c.after, c.category)
                if key not in seen:
                    seen.add(key)
                    out.append(c)
        return out
