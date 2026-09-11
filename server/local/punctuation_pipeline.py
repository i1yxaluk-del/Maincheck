"""Layout-aware and structural punctuation processing for official Russian."""
from __future__ import annotations

import difflib
import re

from decision_engine import EditCandidate
from reasoning_cascade import ReasoningCascade
from safe_diff import diff_candidates
from segmentation import split_sentences, strip_enumeration

TOKEN_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+|[^\sА-Яа-яЁёA-Za-z0-9]")
SOFT_BREAK_RE = re.compile(r"(?<!\n)\n(?!\n)")
WORD_RE = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*")
WORD = r"[А-Яа-яЁё]+"
ORG_RE = re.compile(
    rf"(?P<participle>{WORD})\s+с\s+личн(?:ым|ого|ому|ом)\s+состав(?:ом|а|у|е)"
    rf"(?P<badcomma>\s*,)(?P<space>\s+)"
    rf"(?P<org>(?:ООО|АО|ПАО|ФГБУ|ФГУП)\s*«[^»]+»)"
    rf"(?P<gap>\s+)(?P<next>{WORD})",
    re.IGNORECASE,
)
PREPOSITIONS = {
    "в", "во", "на", "по", "для", "с", "со", "из", "от", "при",
    "к", "ко", "о", "об", "обо", "под", "над", "между", "через",
}
CONJUNCTIONS = {"и", "или", "либо", "а также"}
FINITE_POS = {"VERB", "INFN", "PRTS"}


def collapse_soft_breaks(text: str) -> str:
    return SOFT_BREAK_RE.sub(" ", text)


def restore_soft_breaks(original: str, corrected: str) -> str:
    breaks = [m.start() for m in SOFT_BREAK_RE.finditer(original)]
    if not breaks:
        return corrected
    src = [(m.group(0), m.start()) for m in TOKEN_RE.finditer(original)]
    dst = [(m.group(0), m.start()) for m in TOKEN_RE.finditer(corrected)]
    if not src or not dst:
        return corrected
    matcher = difflib.SequenceMatcher(
        None, [t[0].casefold() for t in src], [t[0].casefold() for t in dst],
        autojunk=False,
    )
    index_map: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for shift in range(i2 - i1):
                index_map[i1 + shift] = j1 + shift
    insertions: list[int] = []
    for position in breaks:
        right = next((i for i, (_, start) in enumerate(src) if start > position), None)
        if right is None:
            continue
        mapped = index_map.get(right)
        if mapped is not None:
            insertions.append(dst[mapped][1])
    result = corrected
    for position in sorted(set(insertions), reverse=True):
        left = position
        while left > 0 and result[left - 1] in " \t":
            left -= 1
        result = result[:left] + "\n" + result[position:]
    return result


class OfficePunctuationRules:
    """High-precision rules for official documents with company names."""

    def __init__(self, morphology) -> None:
        self.morph = morphology

    def candidates(self, text: str) -> list[EditCandidate]:
        out: list[EditCandidate] = []
        for match in ORG_RE.finditer(text):
            participle = match.group("participle")
            following = match.group("next")
            if not any(p.tag.POS == "PRTF" for p in self.morph.attributive_parses(participle)):
                continue
            if not self.morph.attributive_parses(following):
                continue
            comma_pos = match.start("badcomma") + match.group("badcomma").rfind(",")
            out.append(EditCandidate(
                ",", "", 0.995, "rule-punctuation",
                "запятая разрывает дополнение с наименованием организации",
                start=comma_pos,
            ))
            org = match.group("org")
            out.append(EditCandidate(
                org, org + ",", 0.995, "rule-punctuation",
                "закрытие причастного оборота перед определяемым существительным",
                start=match.start("org"),
            ))
        return out


class StructuralPunctuationRules:
    """Remove commas that split one homogeneous nominal construction.

    The rule is lexical-independent.  It requires a complete structural proof:
    a nominal member, its genitive complement, a following prepositional
    modifier without a predicate, and a coordinated nominal member in the same
    case.  This covers a productive official-prose error while avoiding a broad
    and unsafe "comma before preposition" heuristic.
    """

    def __init__(self, morphology) -> None:
        self.morph = morphology

    @staticmethod
    def _cases(parses) -> set[str]:
        return {str(p.tag.case) for p in parses if getattr(p.tag, "case", None)}

    def _has_finite(self, words) -> bool:
        for word in words:
            parses = self.morph.known_parses(word.group(0))
            if any(getattr(p.tag, "POS", None) in FINITE_POS for p in parses):
                return True
        return False

    def candidates(self, text: str) -> list[EditCandidate]:
        words = list(WORD_RE.finditer(text))
        if len(words) < 6:
            return []
        out: list[EditCandidate] = []
        for comma in re.finditer(",", text):
            left_index = next((i for i in range(len(words) - 1, -1, -1)
                               if words[i].end() <= comma.start()), None)
            right_index = next((i for i, word in enumerate(words)
                                if word.start() >= comma.end()), None)
            if left_index is None or right_index is None or right_index == 0:
                continue
            if words[right_index].group(0).casefold() not in PREPOSITIONS:
                continue
            # Require the comma to be adjacent to the nominal phrase modulo
            # whitespace/visual wraps.
            if text[words[left_index].end():comma.start()].strip():
                continue
            if text[comma.end():words[right_index].start()].strip():
                continue
            complement_cases = self._cases(self.morph.noun_parses(words[left_index].group(0)))
            if not complement_cases or not (complement_cases & {"gent", "datv", "ablt", "accs"}):
                continue

            head = None
            head_cases: set[str] = set()
            for item in reversed(words[max(0, left_index - 6):left_index]):
                cases = self._cases(self.morph.noun_parses(item.group(0))) & {"nomn", "accs"}
                if cases:
                    head, head_cases = item, cases
                    break
            if head is None:
                continue

            conjunction_index = None
            for i in range(right_index + 2, min(len(words), right_index + 11)):
                if words[i].group(0).casefold() in {"и", "или", "либо"}:
                    conjunction_index = i
                    break
            if conjunction_index is None:
                continue
            between = words[right_index + 1:conjunction_index]
            if len(between) < 2 or self._has_finite(between):
                continue

            peer_cases: set[str] = set()
            for item in words[conjunction_index + 1:min(len(words), conjunction_index + 5)]:
                peer_cases |= self._cases(self.morph.noun_parses(item.group(0))) & {"nomn", "accs"}
            if not (head_cases & peer_cases):
                continue
            out.append(EditCandidate(
                ",", "", 0.99, "rule-punctuation-structure",
                "запятая разрывает однородную именную конструкцию перед зависимым оборотом",
                start=comma.start(),
            ))
        return out


class LayoutAwareReasoningCascade(ReasoningCascade):
    async def candidates(self, text: str, context: str = "") -> list[EditCandidate]:
        if not self.enabled:
            return []
        out: list[EditCandidate] = []
        for segment in [strip_enumeration(s) for s in split_sentences(text)][:self.max_sentences]:
            model_text = collapse_soft_breaks(segment.text)
            try:
                corrected_flat = await self.correct(model_text, context)
                corrected = restore_soft_breaks(segment.text, corrected_flat)
            except Exception:
                self._failures += 1
                continue
            out.extend(diff_candidates(
                segment.text, corrected, "russian-gec-reasoning", 0.86,
                offset=segment.start,
            ))
        return out
