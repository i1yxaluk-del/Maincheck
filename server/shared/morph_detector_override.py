"""Targeted production override for high-confidence adjective+noun agreement.

The legacy MorphDetector intentionally skips every ``PREP + adjective + noun``
sequence to suppress discourse-marker false positives such as ``при этом
количество``. That heuristic is too broad: real attributive phrases such as
``после ночных наряда`` are skipped as well.

This module adds a conservative second pass for that exact blind spot:
only a real adjective/participle followed by a known noun is considered, and
only when the BEST pymorphy3 parses disagree on number/case/gender. Pronoun
patterns such as ``при этом количестве`` remain excluded.
"""

from __future__ import annotations

import logging
import os

from shared import morph_detector as _md

_log = logging.getLogger("ai_suggester.morph_detector_override")


def _install() -> None:
    """Patch MorphDetector once at package import time."""
    if getattr(_md, "_STRICT_AGREEMENT_OVERRIDE_INSTALLED", False):
        return

    original = _md.MorphDetector.detect_adj_noun_disagreements

    def strict_detect(self, raw_text: str, parsed_doc=None):  # noqa: ANN001
        errors = list(original(self, raw_text, parsed_doc))
        existing = {(e.offset, e.length, e.before, e.suggestion) for e in errors}

        if not getattr(self, "available", False) or not getattr(self, "_morph", None):
            return errors

        words = list(_md._iter_words_with_offsets(raw_text))
        for i in range(1, len(words) - 1):
            _, prev_word = words[i - 1]
            offset_a, word_a = words[i]
            offset_b, word_b = words[i + 1]

            # This pass is only for the legacy broad-preposition skip.
            if not self._is_preposition(prev_word):
                continue
            if not self._is_adj_or_participle(word_a):
                continue
            if self._is_ordinal_numeral(word_a):
                continue
            if self._is_noun(word_a):
                continue
            # Keep pronoun/discourse-marker patterns (``при этом количестве``)
            # out of the strict pass.
            parses_a = self._morph.parse(word_a)
            if not any("ADJF" in str(p.tag) or "PRTF" in str(p.tag) for p in parses_a):
                continue
            if not self._is_noun(word_b):
                continue
            if not self._is_known_word(word_a) or not self._is_known_word(word_b):
                continue

            # Participles may govern an instrumental agent rather than agree
            # with the following noun: ``проводимых подразделениями``.
            if self._is_participle(word_a) and self._can_be_instrumental(word_b):
                continue

            adj_parse = parses_a[0]
            noun_parses = self._morph.parse(word_b)
            if not noun_parses:
                continue
            noun_parse = noun_parses[0]

            # High-confidence guard: require a mismatch in the best parses.
            mismatch = False
            if adj_parse.tag.number and noun_parse.tag.number:
                mismatch |= adj_parse.tag.number != noun_parse.tag.number
            if adj_parse.tag.case and noun_parse.tag.case:
                mismatch |= adj_parse.tag.case != noun_parse.tag.case
            if adj_parse.tag.gender and noun_parse.tag.gender:
                mismatch |= adj_parse.tag.gender != noun_parse.tag.gender
            if not mismatch:
                continue

            suggestion = self._suggest_noun_form(
                word_b,
                adj_parse.tag.number,
                adj_parse.tag.case,
            )
            if not suggestion or suggestion == word_b:
                continue

            key = (offset_b, len(word_b), word_b, suggestion)
            if key in existing:
                continue

            errors.append(_md.GrammarError(
                offset=offset_b,
                length=len(word_b),
                before=word_b,
                suggestion=suggestion,
                kind="adj_noun",
                explanation=(
                    "согласование существительного с прилагательным по числу/падежу"
                ),
            ))
            existing.add(key)
            _log.info(
                "StrictAgreement: добавлено «%s» → «%s» после предлога «%s»",
                word_b,
                suggestion,
                prev_word,
            )

        errors.sort(key=lambda e: (e.offset, e.length))
        return errors

    _md.MorphDetector.detect_adj_noun_disagreements = strict_detect
    _md._STRICT_AGREEMENT_OVERRIDE_INSTALLED = True


# Allow a controlled opt-out for emergency rollback.
if os.getenv("MORPH_DETECTOR_STRICT_AGREEMENT", "true").lower() in (
    "1", "true", "yes", "on",
):
    _install()
