from decision_engine import DecisionEngine
from local_rules import LocalRuleEngine


TEXT = (
    "План профессиональной служебная и физической подготовки на 2026 учебную годов "
    "требует, корректировки (штабная тренировка спланирована в один день с "
    "тактико-специальным занятием с элементами командно-штабной тренировки по одна и той же теме);"
)

SHOOTING_TEXT = (
    "неправильно определяется оценка за стрельбу при выполнении сотрудником несколького "
    "упражнений стрельб, из одного или нескольких вида оружия (вооружения)"
)


def test_local_rules_find_all_regression_targets():
    rules = LocalRuleEngine()
    candidates = rules.candidates(TEXT)
    pairs = {(c.before, c.after) for c in candidates}
    assert ("служебная", "служебной") in pairs
    assert ("2026 учебную годов", "2026 учебный год") in pairs
    assert ("по одна", "по одной") in pairs
    assert ("требует, ", "требует ") in pairs


def test_local_rules_apply_without_llm():
    rules = LocalRuleEngine()
    corrected, accepted = DecisionEngine(min_confidence=0.55, max_changes=8).apply(TEXT, rules.candidates(TEXT))
    assert "служебной" in corrected
    assert "2026 учебный год" in corrected
    assert "по одной и той же теме" in corrected
    assert "требует корректировки" in corrected
    assert len(accepted) >= 4


def test_quantifier_rules_find_shooting_errors():
    pairs = {(c.before, c.after) for c in LocalRuleEngine().candidates(SHOOTING_TEXT)}
    assert ("несколького упражнений", "нескольких упражнений") in pairs
    assert ("вида", "видов") in pairs


def test_quantifier_rules_apply_without_llm():
    corrected, accepted = DecisionEngine(min_confidence=0.55, max_changes=8).apply(
        SHOOTING_TEXT, LocalRuleEngine().candidates(SHOOTING_TEXT)
    )
    assert "нескольких упражнений" in corrected
    assert "одного или нескольких видов оружия" in corrected
    assert len(accepted) >= 2


def test_local_rules_do_not_rewrite_hyphenated_modifier():
    text = "служебно-боевой деятельностей"
    pairs = {(c.before, c.after) for c in LocalRuleEngine().candidates(text)}
    assert ("деятельностей", "деятельности") not in pairs
