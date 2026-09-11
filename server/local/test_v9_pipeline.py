"""Регрессии архитектуры пайплайна v9.

Проверяются инварианты, нарушение которых в v8 привело к тому, что
служба «не находила ошибок»: деструктивный grammar-gate, мёртвый
антигаллюцинационный фильтр, потеря правок неуникальных фрагментов и
отбрасывание ответов моделей из-за служебных обёрток.
"""

from __future__ import annotations

import asyncio

import pytest

from decision_engine import DecisionEngine, EditCandidate
from hybrid_editor import STACKS, HybridRouter
from llm_text import normalize_llm_output, sanitize
from safe_diff import diff_candidates
from segmentation import split_sentences, strip_enumeration
from verification import CandidateArbiter, GenerativeGuard

SENTENCE = "Результаты проверки внесены в соответствующего раздел журнала учета."
FIXED = "Результаты проверки внесены в соответствующий раздел журнала учета."


def make_router(preset: str = "A") -> HybridRouter:
    router = HybridRouter(preset)
    router.sage.enabled = False
    router.gec.enabled = False
    router.languagetool.enabled = False
    return router


# ----------------------------------------------------------------------
# Grammar gate удалён
# ----------------------------------------------------------------------
def test_grammar_gate_is_gone():
    """Ни один публичный или приватный API не должен его возвращать.

    В v8 `_grammar_gate_hit` отключал GEC-специалиста и rescue-модели,
    как только появлялся хоть один кандидат категории `rule-agreement`.
    В журналах прод-сервера это срабатывало на каждом запросе.
    """
    assert not hasattr(HybridRouter, "_grammar_gate_hit")
    assert not hasattr(HybridRouter, "_grammar_gate_enabled")


def test_specialist_runs_even_when_rules_already_found_something():
    router = make_router("X")
    router.gec.enabled = True
    calls: list[str] = []

    async def fake(text: str) -> str:
        calls.append(text)
        return FIXED

    router.gec.correct = fake  # type: ignore[method-assign]
    candidates = asyncio.run(router.candidates(SENTENCE, ""))
    assert calls, "GEC-специалист обязан вызываться независимо от правил"
    assert candidates


def test_rescue_is_skipped_when_a_deterministic_candidate_exists():
    router = make_router("A")
    rescued: list[str] = []

    async def fake_rescue(text, context, segments):
        rescued.append(text)
        return []

    router._rescue_candidates = fake_rescue  # type: ignore[method-assign]
    asyncio.run(router.candidates(SENTENCE, ""))
    assert rescued == []


def test_rescue_runs_when_nothing_was_found():
    router = make_router("A")
    rescued: list[str] = []

    async def fake_rescue(text, context, segments):
        rescued.append(text)
        return []

    router._rescue_candidates = fake_rescue  # type: ignore[method-assign]
    asyncio.run(router.candidates("Проверка проведена в полном объеме.", ""))
    assert rescued


def test_stack_contract():
    assert set(STACKS) == {"A", "B", "X", "Y"}
    assert STACKS["A"].model == "t-tech/T-lite-it-2.1:q4_K_M"
    assert STACKS["B"].model == "hf.co/ai-sage/GigaChat3.1-10B-A1.8B-GGUF:latest"
    assert all("deterministic-first" in info.description for info in STACKS.values())


def test_service_is_usable_without_ollama():
    """Детерминированные стадии не требуют Ollama."""
    router = make_router("X")
    assert router.ollama_required() is False
    candidates = asyncio.run(router.candidates(SENTENCE, ""))
    assert any(c.category.startswith("rule-") for c in candidates)


# ----------------------------------------------------------------------
# Нормализация ответов модели
# ----------------------------------------------------------------------
@pytest.mark.parametrize("raw", [
    "Исправленный текст: " + FIXED,
    "```\n" + FIXED + "\n```",
    "<think>надо поправить падеж</think>\n" + FIXED,
    "«" + FIXED + "»",
    FIXED + "\n\nПояснение: исправлен падеж определения.",
])
def test_model_wrappers_do_not_destroy_the_diff(raw: str):
    """v8 отдавал такой ответ прямо в diff, и правка терялась целиком."""
    assert normalize_llm_output(raw) == FIXED
    assert sanitize(SENTENCE, raw) == FIXED
    assert diff_candidates(SENTENCE, sanitize(SENTENCE, raw), "russian-gec", 0.96)


def test_paraphrase_is_rejected():
    paraphrase = "В документе отражены итоги проверки по соответствующему разделу журнала."
    assert sanitize(SENTENCE, paraphrase) == ""


# ----------------------------------------------------------------------
# Смещения и адресация правок
# ----------------------------------------------------------------------
def test_diff_is_aligned_to_word_boundaries():
    edits = diff_candidates(
        "сотрудником нескольких вида оружия",
        "сотрудником нескольких видов оружия",
        "russian-gec", 0.96,
    )
    assert [(c.before, c.after) for c in edits] == [("вида", "видов")]


def test_non_unique_fragment_is_applied_by_offset():
    """До v9 такая правка молча отбрасывалась как неоднозначная."""
    text = "Проверен раздел и раздел журнала."
    candidate = EditCandidate("раздел", "разделы", 0.99, "rule-agreement", "", start=18)
    corrected, accepted = DecisionEngine(min_confidence=0.5).apply(text, [candidate])
    assert corrected == "Проверен раздел и разделы журнала."
    assert len(accepted) == 1


def test_offset_is_verified_against_the_text():
    text = "Проверен раздел журнала."
    candidate = EditCandidate("раздел", "разделы", 0.99, "rule-agreement", "", start=999)
    corrected, accepted = DecisionEngine(min_confidence=0.5).apply(text, [candidate])
    assert corrected == "Проверен разделы журнала."
    assert len(accepted) == 1


def test_diff_offsets_survive_sentence_segmentation():
    text = "Проверка проведена. Результаты внесены в соответствующего раздел журнала."
    segments = split_sentences(text)
    assert len(segments) == 2
    segment = segments[1]
    assert text[segment.start:segment.end] == segment.text
    edits = diff_candidates(
        segment.text,
        segment.text.replace("соответствующего", "соответствующий"),
        "russian-gec", 0.96, offset=segment.start,
    )
    assert edits
    edit = edits[0]
    assert text[edit.start:edit.start + len(edit.before)] == edit.before


# ----------------------------------------------------------------------
# Сегментация
# ----------------------------------------------------------------------
def test_abbreviations_do_not_split_sentences():
    text = "Нарушен п. 3.2 ст. 15 приказа от 01.02.2026. Приняты меры."
    assert len(split_sentences(text)) == 2


def test_initials_never_split_a_sentence():
    """После инициалов деление неоднозначно, поэтому не делим.

    «Иванов И.И. Замечаний нет» и «Иванов И.И. Петров» неразличимы без
    семантики. Склейка безопаснее разрыва: длинный сегмент лишь
    замедляет модель, а разрыв на месте фамилии искажает её вход.
    """
    text = "Проверку провел Иванов И.И. Замечаний нет."
    assert len(split_sentences(text)) == 1


def test_semicolon_list_items_are_separate_segments():
    text = "выявлено нарушение; принято решение; направлено предписание."
    assert len(split_sentences(text)) == 3


def test_enumeration_marker_is_kept_out_of_the_model_input():
    segment = strip_enumeration(split_sentences("1. Проверка проведена.")[0])
    assert segment.text == "Проверка проведена."
    assert segment.start == 3


# ----------------------------------------------------------------------
# Фильтр галлюцинаций
# ----------------------------------------------------------------------
def test_guard_rejects_case_improvement_of_an_agreeing_word():
    guard = GenerativeGuard()
    text = "Результаты проверки внесены в соответствующий раздел журнала учета."
    candidate = EditCandidate("раздел", "разделе", 0.9, "sage-spell-punc", "", start=text.index("раздел "))
    allowed, reason = guard.allow(candidate, text)
    assert allowed is False
    assert "согласованной" in reason


def test_guard_rejects_lexical_substitution_by_a_model():
    guard = GenerativeGuard()
    text = "Внесены изменения в раздел журнала."
    candidate = EditCandidate("раздел", "разделение", 0.9, "russian-gec", "")
    assert guard.allow(candidate, text)[0] is False


def test_guard_rejects_edits_that_change_numbers():
    guard = GenerativeGuard()
    text = "Нарушен пункт 3 приказа."
    candidate = EditCandidate("пункт 3", "пункт 5", 0.9, "draft_tlite", "")
    assert guard.allow(candidate, text)[0] is False


def test_guard_allows_punctuation_edits():
    guard = GenerativeGuard()
    text = "Сотрудник направлен на объект охраны, для несения службы."
    candidate = EditCandidate("охраны, для", "охраны для", 0.9, "sage-spell-punc", "")
    assert guard.allow(candidate, text)[0] is True


def test_guard_allows_typo_fixes_for_out_of_dictionary_words():
    guard = GenerativeGuard()
    text = "Проведен анализ состояния законости."
    candidate = EditCandidate("законости", "законности", 0.9, "sage-spell-punc", "")
    assert guard.allow(candidate, text)[0] is True


def test_guard_does_not_touch_deterministic_candidates():
    guard = GenerativeGuard()
    text = "Результаты внесены в соответствующего раздел журнала."
    candidate = EditCandidate("соответствующего", "соответствующий", 0.98, "rule-agreement", "")
    assert guard.allow(candidate, text)[0] is True


def test_guard_is_wired_into_the_decision_engine():
    text = "Результаты проверки внесены в соответствующий раздел журнала учета."
    candidate = EditCandidate("раздел", "разделе", 0.9, "sage-spell-punc", "")
    engine = DecisionEngine(min_confidence=0.5, guard=GenerativeGuard())
    corrected, accepted = engine.apply(text, [candidate])
    assert corrected == text
    assert accepted == []
    assert engine.rejections


# ----------------------------------------------------------------------
# Арбитраж
# ----------------------------------------------------------------------
def test_solo_generative_inflection_is_demoted_below_the_threshold():
    arbiter = CandidateArbiter()
    merged = arbiter.merge([EditCandidate("вида", "видов", 0.9, "draft_tlite", "", start=0)])
    assert merged[0].confidence < 0.55


def test_two_independent_models_corroborate_each_other():
    arbiter = CandidateArbiter()
    merged = arbiter.merge([
        EditCandidate("вида", "видов", 0.9, "draft_tlite", "", start=0),
        EditCandidate("вида", "видов", 0.9, "russian-gec", "", start=0),
    ])
    assert len(merged) == 1
    assert merged[0].confidence >= 0.55
    assert merged[0].votes == 2


def test_deterministic_rule_outranks_any_model():
    arbiter = CandidateArbiter()
    merged = arbiter.merge([
        EditCandidate("вида", "видов", 0.70, "rule-quantifier", "", start=0),
        EditCandidate("вида", "видов", 0.99, "draft_giga", "", start=0),
    ])
    assert merged[0].category == "rule-quantifier"
    assert merged[0].confidence >= 1.0
