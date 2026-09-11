"""Регрессии на согласование и детерминированные правила (v9).

Главный контракт: сервер не имеет права изменять корректный
официально-деловой текст. Первый тест воспроизводит именно тот дефект,
с которым пришёл прод-отчёт 10.09.2026.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from decision_engine import DecisionEngine, EditCandidate
from local_rules import LocalRuleEngine
from morphology import get_morphology
from np_agreement import NounPhraseAgreement
from spellcheck import DictionarySpellChecker
from verification import GenerativeGuard

CASES = Path(__file__).resolve().parent / "eval" / "cases.jsonl"

PRODUCTION_REPORT = (
    "результаты выполнения упражнений стрельб, несвоевременно проставляются в "
    "соответствующий раздел журнала учета профессиональной служебной и физической "
    "подготовки (характерно для 1 и 3 отделов охраны);"
)

pytestmark = pytest.mark.skipif(
    not get_morphology().available, reason="требуется pymorphy3 + pymorphy3-dicts-ru",
)


def load_cases(kind: str) -> list[dict]:
    rows = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row["kind"] == kind]


def correct(text: str) -> tuple[str, list[EditCandidate]]:
    rules = LocalRuleEngine()
    speller = DictionarySpellChecker()
    engine = DecisionEngine(min_confidence=0.55, max_changes=12, guard=GenerativeGuard())
    return engine.apply(text, rules.candidates(text) + speller.candidates(text))


# ----------------------------------------------------------------------
# Прод-дефект
# ----------------------------------------------------------------------
def test_production_report_sentence_is_left_untouched():
    """«в соответствующий раздел» не должно превращаться в «соответствующего раздел».

    Причина дефекта v8: `inflect({"accs", "sing", "masc"})` без
    грамммемы одушевлённости возвращает одушевлённую форму, совпадающую
    с родительным падежом.
    """
    corrected, accepted = correct(PRODUCTION_REPORT)
    assert corrected == PRODUCTION_REPORT
    assert accepted == []


def test_masculine_inanimate_accusative_is_not_broken():
    for text in (
        "Указанные недостатки устранены в установленный срок.",
        "Сотрудник направлен на указанный объект охраны.",
        "Начальник отдела подписал соответствующий приказ.",
        "Соответствующие изменения внесены в раздел журнала учета.",
    ):
        assert correct(text)[0] == text


def test_short_participle_is_not_treated_as_attribute():
    """Краткая форма — сказуемое; «Утвержден план» не «Утверждённого план»."""
    text = "Утвержден план профессиональной служебной и физической подготовки."
    assert correct(text)[0] == text


def test_function_word_is_not_a_noun_phrase_head():
    """«в», «и», «то» имеют NOUN-разборы как названия букв."""
    text = "Указанные недостатки устранены в установленный срок."
    assert LocalRuleEngine().candidates(text) == []


def test_yo_convention_of_the_source_is_preserved():
    """Правка падежа не имеет права дописывать «ё» там, где её не было."""
    text = "Проверка проведена за отчетный период 2026 года."
    corrected, _ = correct(text)
    assert "ё" not in corrected


def test_participial_clause_attaches_to_the_left():
    text = "Материальный ущерб, причиненный подразделению, возмещен в добровольном порядке."
    assert correct(text)[0] == text


def test_predicative_instrumental_is_not_an_attribute():
    text = "Назначенный ответственным сотрудник обеспечил сохранность вверенного имущества."
    assert correct(text)[0] == text


def test_substantivized_pronoun_after_preposition():
    text = "При этом количество должностных преступлений за отчетный период не увеличилось."
    assert correct(text)[0] == text


def test_coordinated_singular_modifiers_with_plural_head():
    text = "Данное нарушение характерно для первого и третьего отделов охраны."
    assert correct(text)[0] == text


def test_impersonal_passive_with_negation_is_not_agreed():
    text = "Нарушений порядка хранения материальных ценностей не выявлено."
    assert correct(text)[0] == text


# ----------------------------------------------------------------------
# Полнота
# ----------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    (
        "неправильно определяется оценка за стрельбу при выполнении сотрудником "
        "несколького упражнений стрельб, из одного или нескольких вида оружия;",
        "неправильно определяется оценка за стрельбу при выполнении сотрудником "
        "нескольких упражнений стрельб, из одного или нескольких видов оружия;",
    ),
    (
        "План профессиональной служебная и физической подготовки на 2026 учебных год "
        "требует, корректировки;",
        "План профессиональной служебной и физической подготовки на 2026 учебный год "
        "требует корректировки;",
    ),
    (
        "Результаты проверки внесены в соответствующего раздел журнала учета.",
        "Результаты проверки внесены в соответствующий раздел журнала учета.",
    ),
    (
        "Согласно утвержденному плана проведена штабная тренировка.",
        "Согласно утвержденному плану проведена штабная тренировка.",
    ),
    (
        "Выявлены отдельные недостатки при ведение журнала учета проверок.",
        "Выявлены отдельные недостатки при ведении журнала учета проверок.",
    ),
    (
        "Проверочное мероприятия проведено в полном объеме.",
        "Проверочное мероприятие проведено в полном объеме.",
    ),
    (
        "Начальником отдела охраны принято меры по устранению выявленных нарушений.",
        "Начальником отдела охраны приняты меры по устранению выявленных нарушений.",
    ),
    (
        "Проведен анализ состояния служебной дисциплины и законости.",
        "Проведен анализ состояния служебной дисциплины и законности.",
    ),
    (
        "Сотрудники прошли проверку знаний нормативых правовых актов.",
        "Сотрудники прошли проверку знаний нормативных правовых актов.",
    ),
])
def test_known_error_classes_are_corrected(text: str, expected: str):
    assert correct(text)[0] == expected


# ----------------------------------------------------------------------
# Полный корпус
# ----------------------------------------------------------------------
def test_no_false_positive_on_the_whole_clean_corpus():
    """Ключевой CI-гейт: ни одно корректное предложение не изменено."""
    damaged = []
    for case in load_cases("clean"):
        corrected, accepted = correct(case["text"])
        if corrected != case["text"]:
            damaged.append((case["id"], corrected, [(c.before, c.after, c.category) for c in accepted]))
    assert damaged == []


def test_deterministic_recall_does_not_regress():
    exact = sum(1 for case in load_cases("error") if correct(case["text"])[0] == case["expected"])
    assert exact >= 16, f"точных исправлений {exact} из {len(load_cases('error'))}"


# ----------------------------------------------------------------------
# Компоненты
# ----------------------------------------------------------------------
def test_agreement_is_symmetric_about_ambiguity():
    """При омонимии предпочтение всегда у исходного текста."""
    morph = get_morphology()
    assert morph.pair_agrees("соответствующий", "раздел") is True
    assert morph.pair_agrees("соответствующего", "раздел") is False


def test_np_agreement_reports_offsets():
    text = "Результаты проверки внесены в соответствующего раздел журнала учета."
    edits = NounPhraseAgreement().detect(text)
    assert edits
    start, end, before, after, _ = edits[0]
    assert text[start:end] == before
    assert after == "соответствующий"


def test_spellcheck_requires_an_unambiguous_dictionary_variant():
    speller = DictionarySpellChecker()
    # Нестандартная форма «подразделенья» (помета V-be) не конкурирует.
    edits = speller.candidates("Копия акта вручена руководителю проверяемого подразделеня.")
    assert [(c.before, c.after) for c in edits] == [("подразделеня", "подразделения")]


def test_spellcheck_ignores_abbreviations_and_proper_nouns():
    speller = DictionarySpellChecker()
    assert speller.candidates("Проверку провел ФСВНГ совместно с Ивановвым Сергеем.") == []
