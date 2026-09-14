from decision_engine import DecisionEngine, EditCandidate
from morphology import get_morphology
from v16_rules import V16RuleExtension
from v16_safety import suppress_unsafe_solo_punctuation


def apply_rules(text: str):
    candidates = V16RuleExtension(get_morphology()).candidates(text)
    return DecisionEngine(min_confidence=0.55).apply(text, candidates)


def test_process_government_with_visual_breaks():
    text = "выявлены\nнарушения порядка ведений\nучета учебного гранатометания"
    corrected, accepted = apply_rules(text)
    assert corrected == "выявлены\nнарушения порядка ведения\nучета учебного гранатометания"
    assert [c.category for c in accepted] == ["rule-process-government"]


def test_countable_process_plural_is_not_rewritten():
    text = "Установлен порядок чередований цветов."
    assert apply_rules(text)[0] == text


def test_modifier_chain_fixes_single_outlier_across_visual_break():
    text = ("актуализации сведений о регистрационных\nданных несъемных жесткого\n"
            "магнитных дисков")
    corrected, accepted = apply_rules(text)
    assert corrected.endswith("данных несъемных жестких\nмагнитных дисков")
    assert [c.category for c in accepted] == ["rule-modifier-chain-agreement"]


def test_correct_modifier_chain_is_preserved():
    text = "данных несъемных жестких магнитных дисков"
    corrected, accepted = apply_rules(text)
    assert corrected == text
    assert not accepted


def test_two_modifiers_without_majority_are_not_guessed():
    text = "сведения о жесткого магнитных дисках"
    corrected, accepted = apply_rules(text)
    assert corrected == text
    assert not accepted


def test_document_section_uses_locative_with_stative_predicate():
    text = ("в\nраздел 2 «Учебные предметы» журнала\nучета профессиональной служебной и\n"
            "физической подготовки, не выставляется\nобщая оценка")
    corrected, accepted = apply_rules(text)
    assert corrected.startswith("в\nразделе 2")
    assert any(c.category == "rule-document-locative" for c in accepted)


def test_direction_into_section_stays_accusative():
    text = "В раздел 2 внесены сведения об оценках."
    assert apply_rules(text)[0] == text


def test_comma_before_impersonal_predicate_is_removed():
    text = ("В период с 1 по 2 июля\n2026\nгода в управлении Центра, принято участие\n"
            "в учебно-методических занятиях.")
    corrected, accepted = apply_rules(text)
    assert "в управлении Центра принято участие" in corrected.replace("\n", " ")
    assert [c.category for c in accepted] == ["rule-predicate-boundary"]


def test_required_comma_before_responsible_modifier_is_preserved():
    text = "Помощь оказана лицам, ответственным за подготовку."
    assert apply_rules(text)[0] == text


def test_solo_sage_comma_addition_is_suppressed():
    candidate = EditCandidate(
        "профессиональной служебной", "профессиональной, служебной",
        0.8, "sage-spell-punc", sources=("sage-spell-punc",), start=0,
    )
    assert suppress_unsafe_solo_punctuation([candidate])[0].confidence < 0.55


def test_asymmetric_quote_replacement_is_suppressed():
    candidate = EditCandidate("«", '"', 0.8, "sage-spell-punc", start=10)
    assert suppress_unsafe_solo_punctuation([candidate])[0].confidence < 0.55


def test_rupunct_comma_addition_remains_available():
    candidate = EditCandidate(
        "слово", "слово,", 0.9, "sage-rupunct",
        sources=("sage-rupunct",), start=0,
    )
    assert suppress_unsafe_solo_punctuation([candidate])[0].confidence == 0.9


def test_model_comma_deletion_remains_available():
    candidate = EditCandidate("Центра,", "Центра", 0.8, "sage-spell-punc", start=0)
    assert suppress_unsafe_solo_punctuation([candidate])[0].confidence == 0.8
