"""Tests for v1.8a morph_detector — детектор грамматических ошибок.

Покрывает три класса детекций:
  * numeral_noun (число числительного ≠ число существительного)
  * adj_noun (рассогласование прилагательного/причастия с существительным)
  * oov (слово не найдено в словаре pymorphy3, не аббревиатура и не имя)

Также покрывает дедупликацию, whitelist через user_dict и no-op
поведение когда pymorphy3 недоступен.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Подключаем shared/ к sys.path как в основном коде
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "server"))

from shared.morph_detector import (  # noqa: E402
    GrammarError,
    MorphDetector,
    _looks_like_abbreviation,
)


@pytest.fixture(scope="module")
def detector() -> MorphDetector:
    """Single instance — pymorphy3 init дорогой."""
    d = MorphDetector()
    if not d.available:
        pytest.skip("pymorphy3 не установлен, детектор недоступен")
    return d


# ─── numeral_noun детектор ───────────────────────────────────────────


def test_numeral_noun_kvartalakh(detector: MorphDetector):
    """Главный prod-кейс: «во 2-м кварталах» — sing numr + plur noun."""
    errs = detector.detect_errors(
        "Во 2-м кварталах планируется направление материалов проверки."
    )
    befores = [e.before for e in errs]
    assert "кварталах" in befores
    err = next(e for e in errs if e.before == "кварталах")
    assert err.kind == "numeral_noun"
    assert err.suggestion == "квартале"


def test_numeral_noun_5kh_sluchayakh_no_error(detector: MorphDetector):
    """В 5-х (plur) случаях (plur) — согласование, ошибки нет."""
    errs = detector.detect_errors("В 5-х случаях наблюдается рост.")
    assert errs == []


def test_numeral_noun_pervom_polugodii_no_error(detector: MorphDetector):
    """В первом (sing) полугодии (sing) — согласование, ошибки нет."""
    errs = detector.detect_errors("В первом полугодии прибыль выросла.")
    assert errs == []


# ─── adj_noun детектор ───────────────────────────────────────────────


def test_adj_noun_meropriyatie(detector: MorphDetector):
    """«Проверочное (sing.neut.nomn) мероприятия (sing.gent или plur)» — disagreement."""
    errs = detector.detect_errors(
        "Проверочное мероприятия по факту допущенных нарушений продолжаются."
    )
    befores = [e.before for e in errs]
    assert "мероприятия" in befores
    err = next(e for e in errs if e.before == "мероприятия")
    assert err.kind == "adj_noun"
    assert err.suggestion == "мероприятие"


def test_adj_noun_dokumenta(detector: MorphDetector):
    """«Подписанные (plur) документа (sing.gent)» — disagreement."""
    errs = detector.detect_errors("Подписанные документа отправлены.")
    befores = [e.before for e in errs]
    assert "документа" in befores


def test_adj_noun_short_form_no_false_positive(detector: MorphDetector):
    """«указаны (PRTS plur) работы (NOUN plur)» — краткое причастие
    не склоняется по падежам, проверяем только число — agreement OK."""
    errs = detector.detect_errors("В акте КС-2 указаны работы.")
    assert errs == []


def test_adj_noun_documents_signed_no_error(detector: MorphDetector):
    """«Документы (plur) подписаны (PRTS plur)» — agreement OK."""
    errs = detector.detect_errors("Документы подписаны вчера.")
    assert errs == []


# ─── oov детектор ────────────────────────────────────────────────────


def test_oov_remontova(detector: MorphDetector):
    """Главный prod-кейс: «капитальных ремонтова» — выдуманное слово."""
    errs = detector.detect_errors("При проведении капитальных ремонтова в ЦСН.")
    # Ремонтова детектируется как adj_noun (disagree с «капитальных»);
    # всё равно появится в результатах. ЦСН — аббревиатура, не флаг.
    befores = [e.before for e in errs]
    assert "ремонтова" in befores
    assert "ЦСН" not in befores


def test_oov_byuryufyl_unknown_word(detector: MorphDetector):
    """Полностью бессмысленное слово — флагуется как OOV."""
    errs = detector.detect_errors("Это бярюфьл текст.")
    befores = [e.before for e in errs]
    assert "бярюфьл" in befores
    err = next(e for e in errs if e.before == "бярюфьл")
    assert err.kind == "oov"


def test_oov_abbreviation_csn_safe(detector: MorphDetector):
    """ЦСН — аббревиатура, не должна флагаться как OOV."""
    errs = detector.detect_errors("Сотрудники ЦСН ВО провели проверку.")
    befores = [e.before for e in errs]
    assert "ЦСН" not in befores
    assert "ВО" not in befores


def test_oov_abbreviation_with_digit_safe(detector: MorphDetector):
    """КС-2 — аббревиатура с цифрой, не должна флагаться как OOV."""
    errs = detector.detect_errors("В акте КС-2 указаны работы.")
    befores = [e.before for e in errs]
    assert "КС-2" not in befores


def test_oov_proper_name_safe(detector: MorphDetector):
    """Иванов — фамилия (Surn-парс), не флагуется."""
    errs = detector.detect_errors("Иванов И.И. подал отчет.")
    befores = [e.before for e in errs]
    assert "Иванов" not in befores


def test_oov_whitelist_filters(detector: MorphDetector):
    """Whitelist (user_dict) исключает слово из OOV-проверки."""
    errs_no_wl = detector.detect_errors("Это бярюфьл текст.")
    assert any(e.before == "бярюфьл" for e in errs_no_wl)

    errs_wl = detector.detect_errors(
        "Это бярюфьл текст.",
        whitelist=frozenset({"бярюфьл"}),
    )
    assert not any(e.before == "бярюфьл" for e in errs_wl)


# ─── эвристика аббревиатур ───────────────────────────────────────────


def test_looks_like_abbreviation_uppercase():
    assert _looks_like_abbreviation("ЦСН")
    assert _looks_like_abbreviation("МЧС")
    assert _looks_like_abbreviation("ВО")
    assert _looks_like_abbreviation("FBI")


def test_looks_like_abbreviation_with_digit_dash():
    assert _looks_like_abbreviation("КС-2")
    assert _looks_like_abbreviation("ГОСТ-12345")


def test_looks_like_abbreviation_rejects_normal():
    assert not _looks_like_abbreviation("Иванов")
    assert not _looks_like_abbreviation("документ")
    assert not _looks_like_abbreviation("работа")


def test_looks_like_abbreviation_rejects_short():
    assert not _looks_like_abbreviation("")
    assert not _looks_like_abbreviation("А")  # одна буква


# ─── интеграционные кейсы ────────────────────────────────────────────


def test_full_prod_case_meropriyatie_kvartalakh(detector: MorphDetector):
    """Полный prod-кейс из extension-теста: ловим обе ошибки."""
    text = (
        "Проверочное мероприятия по факту допущенных нарушений, "
        "при проведении капитального ремонта в ЦСН и установлению "
        "лиц, причастных к их совершению продолжаются. Во 2-м "
        "кварталах планируется направление материалов проверки."
    )
    errs = detector.detect_errors(text)
    befores = {e.before for e in errs}
    assert "мероприятия" in befores
    assert "кварталах" in befores
    # ЦСН — аббрев., не должна флагаться
    assert "ЦСН" not in befores


def test_no_false_positives_on_clean_admin_text(detector: MorphDetector):
    """На чистом админ-тексте детектор не должен поднимать ложно-положительных."""
    text = (
        "Проведенными мероприятиями во взаимодействии с отделом УФ установлены "
        "факты завышения стоимости выполненной работы. Подразделению были "
        "выделены средства в размере более 2 млн рублей."
    )
    errs = detector.detect_errors(text)
    assert errs == []


def test_dedupe_same_offset(detector: MorphDetector):
    """Если та же позиция flagуется и numeral_noun, и adj_noun —
    в выходе должна остаться только одна (приоритет numeral_noun)."""
    # numeral + adj + noun: «2-м (numr) важных (adj) делах (noun.plur)» —
    # adj согласован с noun (plur), а numr (sing) — нет.
    errs = detector.detect_errors("Во 2-м кварталах было решено.")
    befores = [e.before for e in errs]
    # «кварталах» ловится только numeral_noun (одна запись)
    assert befores.count("кварталах") == 1


def test_no_op_when_no_pymorphy(monkeypatch):
    """Если pymorphy3 не загружен, детектор no-op."""
    d = MorphDetector.__new__(MorphDetector)
    d._morph = None
    assert not d.available
    assert d.detect_errors("Какой-то текст с ошибками") == []
    assert d.detect_numeral_noun_disagreements("Во 2-м кварталах") == []
    assert d.detect_adj_noun_disagreements("Подписанные документа") == []
    assert d.detect_oov_words("бярюфьл") == []


# ─── GrammarError dataclass ──────────────────────────────────────────


def test_grammar_error_to_change_line():
    err = GrammarError(
        offset=10, length=9, before="кварталах", suggestion="квартале",
        kind="numeral_noun",
        explanation="согласование существительного с числительным по числу",
    )
    line = err.to_change_line(3)
    assert line == (
        "3. «кварталах» → «квартале» | "
        "согласование существительного с числительным по числу"
    )


def test_empty_text_no_errors(detector: MorphDetector):
    assert detector.detect_errors("") == []
    assert detector.detect_errors(" \n\t ") == []


# ─── v1.8.1 регрессионные тесты ──────────────────────────────────────


def test_v181_no_fp_chain_with_oov_neighbour(detector: MorphDetector):
    """v1.8.1 регрессия: «капитальных ремонтова помещений административного
    здания» — «ремонтова» — OOV. Раньше детектор парсил его как ADJS через
    FakeDictionary и выдавал ДВА FP: «ремонтова → ремонтовых» (adj_noun) и
    «помещений → помещения» (adj_noun, ремонтова в роли prev-adj). Теперь
    OOV-сосед пропускается в adj_noun/numeral_noun, и оба FP исчезают.
    «ремонтова» по-прежнему ловится OOV-детектором.
    """
    text = (
        "за выполнением капитальных ремонтова помещений "
        "административного здания «ЦСН ВО»"
    )
    errs = detector.detect_errors(text)
    befores = [e.before for e in errs]
    # «помещений» НЕ флагуется (был FP до v1.8.1)
    assert "помещений" not in befores
    # «ремонтова» ловится OOV-детектором (а не adj_noun)
    kinds_by_before = {e.before: e.kind for e in errs}
    assert kinds_by_before.get("ремонтова") == "oov"


def test_v181_oov_word_with_fake_surn_parse_flagged(detector: MorphDetector):
    """v1.8.1: выдуманные слова на «-ова», «-ин», «-ский» получают
    FakeDictionary-парс с Surn-тегом, но is_known=False. Должны
    флагаться как OOV, не пропускаться как «фамилия».
    """
    errs = detector.detect_errors("При проведении капитальных ремонтова в ЦСН.")
    befores = {e.before for e in errs}
    assert "ремонтова" in befores
    # Параллельно: реальная фамилия Иванов (is_known=True, Surn) НЕ флагуется
    errs2 = detector.detect_errors("Подписано Ивановым.")
    befores2 = {e.before for e in errs2}
    assert "Ивановым" not in befores2


def test_v181_no_fp_on_chain_with_real_noun_genitive(detector: MorphDetector):
    """v1.8.1: «административного здания» — adj.sing.gent + noun.sing.gent
    (согласованы). Детектор не должен флагать связку, и не должен флагать
    «помещений» через «административного» — оно НЕ соседний noun.
    """
    text = "помещений административного здания"
    errs = detector.detect_errors(text)
    befores = [e.before for e in errs]
    assert "помещений" not in befores
    assert "здания" not in befores


# ─── v1.8.5 регрессионные тесты ──────────────────────────────────────


def test_v185_no_fp_participle_with_instrumental_agent(detector: MorphDetector):
    """v1.8.5 прод-кейс (05.05.2026): корректное «проводимых подразделениями»
    флагалось как disagreement и «исправлялось» в «проводимых подразделений»,
    что грамматически неверно.

    Причастие «проводимых» (gen.pl) согласуется с upstream-головой
    («мероприятий ... преступлений», gen.pl), а «подразделениями» — это
    его агенс в творительном падеже (instrumental). Это норма русского
    языка для пассивных причастий: «работа, выполненная Ивановым»,
    «решение, принятое комиссией», «преступление, совершённое лицом».
    """
    text = (
        "УОПМ в целях повышения эффективности мероприятий по профилактике, "
        "предупреждению, выявлению и пресечению преступлений, проводимых "
        "подразделениями собственной безопасности, обобщены и структурированы "
        "сведения о преступлениях, совершенных должностными лицами "
        "подразделений вневедомственной охраны и разрешительной работы "
        "территориальных органов."
    )
    errs = detector.detect_errors(text)
    befores = [e.before for e in errs]
    # Главное: НЕТ FP на «подразделениями» (агенс причастия, не disagreement)
    assert "подразделениями" not in befores, (
        f"FP regression: «подразделениями» flagged as error: {befores}"
    )
    # Параллельно: «должностными» (instrumental) после «совершенных» —
    # тоже агенс, не должен флагаться
    assert "должностными" not in befores
    assert "лицами" not in befores


def test_v185_participle_with_agent_simple_cases(detector: MorphDetector):
    """v1.8.5: краткие изолированные случаи паттерна «причастие + агенс»."""
    cases = [
        # (text, instrumental_noun_that_should_NOT_be_flagged)
        ("работа, выполненная Ивановым", "Ивановым"),
        ("решение, принятое комиссией", "комиссией"),
        ("документ, подписанный директором", "директором"),
        ("приказ, утверждённый руководителем", "руководителем"),
        ("отчёт, рассмотренный отделом", "отделом"),
    ]
    for text, agent in cases:
        errs = detector.detect_adj_noun_disagreements(text)
        befores = [e.before for e in errs]
        assert agent not in befores, (
            f"v1.8.5 regression for «{text}»: agent «{agent}» wrongly flagged"
        )


def test_v185_participle_attribute_agreement_still_detected(
    detector: MorphDetector,
):
    """v1.8.5: проверка что мы НЕ сломали детектор для реальных
    рассогласований причастий с головой. Пример: «*проведённое мероприятия»
    — «проведённое» (sg.neut.nomn) + «мероприятия» (NOT nomn.neut.sg —
    либо plur либо sg.gent) — disagreement.

    Поскольку «мероприятия» не парсится в творительном (NOUN, neut, plur
    или sg.gent — нет ablt), v1.8.5 skip-правило НЕ срабатывает,
    disagreement по-прежнему ловится.
    """
    errs = detector.detect_adj_noun_disagreements("проведённое мероприятия")
    befores = [e.before for e in errs]
    assert "мероприятия" in befores, (
        "Реальное рассогласование причастие+сущ должно по-прежнему ловиться"
    )


def test_v185_helper_is_participle(detector: MorphDetector):
    """v1.8.5: _is_participle различает причастия и обычные ADJF."""
    # Причастия
    assert detector._is_participle("проводимых")  # PRTF
    assert detector._is_participle("выполненная")  # PRTF
    assert detector._is_participle("совершенных")  # PRTF
    # ADJF — НЕ причастие (для них v1.8.5 skip не применяется)
    assert not detector._is_participle("капитальных")  # ADJF
    assert not detector._is_participle("красный")  # ADJF
    assert not detector._is_participle("новый")  # ADJF


def test_v185_helper_can_be_instrumental(detector: MorphDetector):
    """v1.8.5: _can_be_instrumental корректно определяет творительный."""
    # Творительный падеж
    assert detector._can_be_instrumental("подразделениями")
    assert detector._can_be_instrumental("Ивановым")
    assert detector._can_be_instrumental("комиссией")
    # НЕ творительный (гарантированно)
    assert not detector._can_be_instrumental("подразделений")  # gen.pl
    assert not detector._can_be_instrumental("дом")  # nomn.sg


def test_v185_helper_is_preposition(detector: MorphDetector):
    """v1.8.5: _is_preposition корректно опознаёт предлоги."""
    # Закрытый класс предлогов
    assert detector._is_preposition("при")
    assert detector._is_preposition("в")
    assert detector._is_preposition("на")
    assert detector._is_preposition("о")
    assert detector._is_preposition("по")
    assert detector._is_preposition("над")
    assert detector._is_preposition("под")
    assert detector._is_preposition("от")
    assert detector._is_preposition("из")
    # case-insensitive: «При» в начале предложения
    assert detector._is_preposition("При")
    # НЕ предлоги
    assert not detector._is_preposition("этом")  # NPRO/ADJF
    assert not detector._is_preposition("новый")  # ADJF
    assert not detector._is_preposition("дом")  # NOUN


def test_v185_no_fp_preposition_object_followed_by_noun(detector: MorphDetector):
    """v1.8.5 прод-кейс (05.05.2026): «при этом количество должностных
    преступлений» — корректный текст. «при этом» — discourse marker,
    «этом» — объект предлога «при», и НЕ модифицирует «количество».
    Детектор не должен выдавать FP «количество → количестве».
    """
    text = (
        "Преобладают общеуголовные преступления — 99, при этом количество "
        "должностных преступлений — 47."
    )
    errs = detector.detect_adj_noun_disagreements(text)
    befores = [e.before for e in errs]
    assert "количество" not in befores, (
        f"v1.8.5 regression: «количество» wrongly flagged in «при этом количество»; "
        f"got errs={[(e.before, e.suggestion) for e in errs]}"
    )


def test_v185_no_fp_preposition_demonstrative_pronoun_cases(detector: MorphDetector):
    """v1.8.5: разные паттерны «<предлог> <местоим/прил> <сущ>»."""
    cases = [
        # (текст, существительное которое НЕ должно быть во FP)
        ("Запись о том сотруднике сделана позже.", "сотруднике"),
        ("Решение по тем вопросам отложено.", "вопросам"),
        ("В этом году проведена реформа.", "году"),
        ("О тех проблемах не сообщили.", "проблемах"),
    ]
    for text, target_noun in cases:
        errs = detector.detect_adj_noun_disagreements(text)
        befores = [e.before for e in errs]
        # Эти существительные могут не быть отмечены вообще, но даже если
        # отмечены — это FP. v1.8.5 это исправляет.
        assert target_noun not in befores, (
            f"v1.8.5 regression for «{text}»: «{target_noun}» wrongly flagged"
        )
