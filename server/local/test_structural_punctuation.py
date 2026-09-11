from decision_engine import DecisionEngine
from morphology import get_morphology
from punctuation_pipeline import StructuralPunctuationRules


def apply(text: str) -> str:
    rules = StructuralPunctuationRules(get_morphology())
    result, _ = DecisionEngine(min_confidence=0.55).apply(text, rules.candidates(text))
    return result


def test_removes_comma_splitting_homogeneous_nominal_members():
    text = ("Изучена\nуправленческая роль должностного лица,\n"
            "в организации служебно-боевой деятельности\n"
            "и фактическое положение дел \nв\nподразделениях Центра.")
    assert apply(text) == text.replace("лица,", "лица")


def test_is_lexically_general_not_phrase_specific():
    text = ("Проверены готовность личного состава, к выполнению поставленной задачи "
            "и техническое состояние оборудования.")
    assert apply(text) == text.replace("состава,", "состава")


def test_keeps_comma_when_second_clause_has_predicate():
    text = ("Изучена роль должностного лица, в организации произошли изменения "
            "и сотрудники приступили к работе.")
    assert apply(text) == text


def test_keeps_clarifying_time_phrase():
    text = "Совещание состоялось вечером, в восемь часов, и завершилось вовремя."
    assert apply(text) == text
