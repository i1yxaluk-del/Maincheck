from decision_engine import DecisionEngine
from v10_rules import V10RuleExtension


def apply(text: str) -> str:
    corrected, _ = DecisionEngine(min_confidence=0.5).apply(
        text, V10RuleExtension().candidates(text)
    )
    return corrected


def test_production_case_with_visual_line_wraps():
    source = (
        "Кроме\nтого, имеются факты, заступления\nна суточное дежурство "
        "сотрудников после\nночных\nнаряда, а также одновременное пребывание\n"
        "в\nРОШ и на стрельбах."
    )
    expected = source.replace("факты, заступления", "факты заступления").replace(
        "ночных\nнаряда", "ночных\nнарядов"
    )
    assert apply(source) == expected


def test_number_rule_works_without_line_wraps():
    assert apply("Сотрудники прибыли после ночных наряда.") == (
        "Сотрудники прибыли после ночных нарядов."
    )


def test_nominal_data_chain_is_not_damaged():
    text = "Отчет подготовлен после данных анализа и проверки."
    assert apply(text) == text


def test_fact_enumeration_is_not_damaged():
    text = "Выявлены факты, нарушения и недостатки."
    assert apply(text) == text
