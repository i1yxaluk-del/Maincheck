from decision_engine import DecisionEngine, EditCandidate
from morphology import get_morphology
from v16_safety import suppress_unsafe_solo_punctuation
from v17_rules import V17RuleExtension


def apply_rules(text: str):
    candidates = V17RuleExtension(get_morphology()).candidates(text)
    return DecisionEngine(min_confidence=0.55).apply(text, candidates)


def test_process_after_pri_uses_singular_locative():
    text = ("Указанные факты создают риски при\nнесениях службы по охране общественного\n"
            "порядка и в составе РОШ, а также производстве выплат.")
    corrected, accepted = apply_rules(text)
    assert "при\nнесении службы" in corrected
    assert [c.category for c in accepted] == ["rule-process-after-pri"]


def test_legitimate_repeated_events_stay_plural():
    text = "Вывод подтвержден при повторных чтениях документа."
    assert apply_rules(text)[0] == text


def test_comma_after_course_phrase_is_removed():
    corrected, accepted = apply_rules("В ходе теста, появились ошибки.")
    assert corrected == "В ходе теста появились ошибки."
    assert [c.category for c in accepted] == ["rule-course-predicate-boundary"]


def test_model_cannot_remove_service_abbreviation():
    candidate = EditCandidate(
        "10 СВТ информационные", "10 дискахформационные",
        0.86, "russian-gec", start=3,
    )
    result = suppress_unsafe_solo_punctuation([candidate], "на 10 СВТ информационные бирки")
    assert result[0].confidence < 0.55


def test_model_cannot_fuse_words():
    candidate = EditCandidate(
        "СВТ информационные", "дискахформационные",
        0.86, "russian-gec", start=6,
    )
    assert suppress_unsafe_solo_punctuation([candidate])[0].confidence < 0.55


def test_parallel_participle_agreement_is_preserved():
    text = ("и имеющего специальное звание полиции, проходящего испытание, "
            "изучения его личных и деловых качеств")
    candidate = EditCandidate(
        "имеющего", "имеющее", 0.86, "russian-gec", start=2,
    )
    result = suppress_unsafe_solo_punctuation([candidate], text)
    assert result[0].confidence < 0.55


def test_safe_word_inflection_remains_available():
    candidate = EditCandidate(
        "жесткого", "жестких", 0.86, "russian-gec", start=0,
    )
    assert suppress_unsafe_solo_punctuation([candidate])[0].confidence == 0.86
