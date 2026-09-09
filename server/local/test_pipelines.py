from decision_engine import DecisionEngine, EditCandidate
from pipelines import MorphologyRescue, RetrievalExamples, SurfaceGate, TliteClient, surface_candidates
from pipelines import STACKS


BAD_SOURCE = """Изучена
управленческая роль, должностного
лиц в организации служебно-боевой
деятельностей
и фактическое положение дел

в
подразделениях Центра."""

BAD_F_OUTPUT = """Изучено
управленческая роль, должностных лица
в организации служебно-боевой деятельности
и фактическое положение дел

в
подразделениях Центра."""

# Точный текст из прод-регрессии 2026-09-08 (без переносов строк —
# как реально приходит из LibreOffice-выделения одной строкой).
PROD_REGRESSION_TEXT = (
    "Изучена управленческая роль, должностного лиц в организации "
    "служебно-боевой деятельностей и фактическое положение дел "
    "в подразделениях Центра."
)


def test_decision_engine_exact_occurrence():
    source = "Это важный акт."
    candidate = EditCandidate("важный", "важная", 0.9, "agreement", "test")
    corrected, accepted = DecisionEngine().apply(source, [candidate])
    assert corrected == "Это важная акт."
    assert len(accepted) == 1


def test_declared_model_presets_are_supported():
    assert {"A", "B", "C", "F", "G"}.issubset(STACKS)


def test_decision_engine_rejects_compound_term_substitution():
    source = "в организации служебно-боевой деятельности"
    candidate = EditCandidate("служебно-боевой", "служебно-бытовой", 0.99, "spelling", "test")
    corrected, accepted = DecisionEngine().apply(source, [candidate])
    assert corrected == source
    assert accepted == []


def test_decision_engine_rejects_unverified_valid_inflection():
    source = "Общий пробег транспортного средства за сутки, км и Горючее"
    candidate = EditCandidate("Горючее", "Горючего", 0.99, "model:agreement", "test")
    corrected, accepted = DecisionEngine().apply(source, [candidate])
    assert corrected == source
    assert accepted == []


def test_decision_engine_rejects_unverified_word_split():
    source = "предрейсовый медицинский осмотр"
    candidate = EditCandidate("предрейсовый", "пред рейсовый", 0.99, "model:spelling", "test")
    corrected, accepted = DecisionEngine().apply(source, [candidate])
    assert corrected == source
    assert accepted == []


def test_morphology_rescue_finds_real_local_agreement_error():
    rescue = MorphologyRescue()
    candidates = rescue.candidates(BAD_SOURCE)
    assert any(c.before == "лиц" and c.after == "лица" for c in candidates)


def test_morphology_rescue_finds_real_local_agreement_error_single_line():
    """Тот же кейс, но без переносов строк (реальный прод-инпут из
    LibreOffice-выделения). Регрессия 2026-09-08: stack A нашёл
    «лиц→лица» на этом тексте, stack G — нет, хотя оба используют
    один и тот же детерминированный LocalEditTagger/MorphologyRescue.
    Этот тест фиксирует, что детерминированный слой сам по себе
    находит кандидата стабильно вне зависимости от переносов строк —
    значит расхождение A vs G было в verify()-прослойке, не здесь."""
    rescue = MorphologyRescue()
    candidates = rescue.candidates(PROD_REGRESSION_TEXT)
    assert any(c.before == "лиц" and c.after == "лица" for c in candidates)


def test_morphology_rescue_corrects_modifier_number_in_noun_group():
    rescue = MorphologyRescue()
    candidates = rescue.candidates(PROD_REGRESSION_TEXT)
    assert any(c.before == "должностного" and c.after == "должностных" for c in candidates)


def test_morphology_rescue_rejects_observed_false_positive_context():
    rescue = MorphologyRescue()
    candidates = rescue.candidates(BAD_SOURCE)
    assert not any(c.before == "деятельностей" and c.after == "деятельности" for c in candidates)


def test_morphology_rescue_does_not_touch_a_bare_predicate():
    rescue = MorphologyRescue()
    candidates = rescue.candidates("Изучена управленческая роль.")
    assert not any(c.before == "изучена" and c.after == "изучено" for c in candidates)


def test_surface_diff_rejects_paragraph_reflow():
    source = "строка один.\nстрока два."
    corrected = "строка один. строка два."
    assert surface_candidates(source, corrected) == []


def test_surface_gate_rejects_observed_f_form_changes():
    gate = SurfaceGate()
    assert gate.accept("изучена", "изучено") is False
    assert gate.accept("должностного", "должностных") is False
    assert gate.accept("деятельностей", "деятельности") is False


def test_full_observed_f_output_has_no_admissible_surface_edits():
    gate = SurfaceGate()
    candidates = surface_candidates(BAD_SOURCE, BAD_F_OUTPUT)
    safe = [c for c in candidates if gate.accept(c.before, c.after)]
    assert safe == []


def test_surface_changes_are_bounded():
    edits = surface_candidates("Это текст.", "Это текст,")
    assert all(len(e.before) <= 80 and len(e.after) <= 80 for e in edits)


def test_retrieval_is_optional_and_local():
    retrieval = RetrievalExamples()
    assert hasattr(retrieval, "examples")


# ─── v2.4.1: регрессия на fail-closed баг в TliteClient.verify() (стек G) ──
#
# Прод-инцидент 2026-09-08: на PROD_REGRESSION_TEXT stack A (local +
# generated) нашёл и принял «лиц→лица» (candidates=1, accepted=1). Stack G
# (local → verify) на ТОМ ЖЕ тексте вернул candidates=0. Детерминированный
# слой идентичен для обоих стеков и не зависит от LLM, поэтому единственное
# объяснение — verify() тихо выбросил валидного кандидата при сбое связи с
# Ollama или при непарсящемся JSON, трактуя ЛЮБУЮ проблему коммуникации как
# «LLM сказал нет». Тесты ниже фиксируют исправленное fail-open поведение
# через monkeypatch chat_json (без реального обращения к Ollama).

import asyncio  # noqa: E402


def _tlite_with_mocked_chat_json(payload_or_exc):
    """Создаёт TliteClient с подменённым chat_json: либо возвращает
    заданный dict, либо бросает переданное исключение."""
    client = TliteClient(RetrievalExamples())

    async def fake_chat_json(messages, schema):
        if isinstance(payload_or_exc, Exception):
            raise payload_or_exc
        return payload_or_exc

    client.chat_json = fake_chat_json  # type: ignore[method-assign]
    return client


def test_verify_fail_open_on_network_exception():
    """Сбой связи с Ollama (таймаут/HTTP-ошибка) НЕ должен отбрасывать
    уже найденные детерминированные кандидаты — раньше отбрасывал."""
    client = _tlite_with_mocked_chat_json(TimeoutError("connection timed out"))
    candidates = [EditCandidate("лиц", "лица", 0.96, "agreement", "test")]
    result = asyncio.run(client.verify("текст", candidates))
    assert result == candidates, (
        "verify() должен fail-open на сетевой ошибке и вернуть исходных "
        "кандидатов без изменений, а не пустой список"
    )


def test_verify_fail_open_on_unparseable_json():
    """T-lite вернул содержимое без ожидаемого поля `accept` (например,
    модель ответила не по схеме) — тоже fail-open, не пустой список."""
    client = _tlite_with_mocked_chat_json({"unexpected_field": True})
    candidates = [EditCandidate("лиц", "лица", 0.96, "agreement", "test")]
    result = asyncio.run(client.verify("текст", candidates))
    assert result == candidates


def test_verify_fail_open_on_empty_accept_list():
    """`accept: []` — формально валидный JSON, но пустой массив
    (модель ничего не провалила). Тоже fail-open."""
    client = _tlite_with_mocked_chat_json({"accept": []})
    candidates = [EditCandidate("лиц", "лица", 0.96, "agreement", "test")]
    result = asyncio.run(client.verify("текст", candidates))
    assert result == candidates


def test_verify_fail_open_on_short_accept_list():
    """`accept` короче списка кандидатов — недостающие кандидаты
    остаются (fail-open по каждому непроголосованному элементу),
    явно провалидированные — учитываются штатно."""
    client = _tlite_with_mocked_chat_json({"accept": [True]})
    candidates = [
        EditCandidate("лиц", "лица", 0.96, "agreement", "test"),
        EditCandidate("деятельностей", "деятельности", 0.70, "agreement", "test2"),
    ]
    result = asyncio.run(client.verify("текст", candidates))
    assert result == candidates  # оба остались: первый явно True, второй — fail-open


def test_verify_explicit_false_still_rejects():
    """Главное: явный `accept[i]=false` от модели ПОСЛЕ успешного
    парсинга JSON по-прежнему отклоняет кандидата. Fail-open касается
    только инфраструктурных сбоев, не подменяет содержательный вердикт."""
    client = _tlite_with_mocked_chat_json({"accept": [False]})
    candidates = [EditCandidate("подозрительная", "правка", 0.55, "agreement", "test")]
    result = asyncio.run(client.verify("текст", candidates))
    assert result == []


def test_verify_explicit_mixed_true_false():
    """Смешанный явный вердикт: True остаётся, False отклоняется."""
    client = _tlite_with_mocked_chat_json({"accept": [True, False]})
    candidates = [
        EditCandidate("лиц", "лица", 0.96, "agreement", "ok"),
        EditCandidate("сомнительное", "спорное", 0.50, "agreement", "reject"),
    ]
    result = asyncio.run(client.verify("текст", candidates))
    assert len(result) == 1
    assert result[0].before == "лиц"


def test_verify_empty_candidates_returns_empty_without_calling_llm():
    """Пустой вход — пустой выход, без обращения к chat_json вообще."""
    called = {"n": 0}
    client = TliteClient(RetrievalExamples())

    async def fake_chat_json(messages, schema):
        called["n"] += 1
        return {"accept": []}

    client.chat_json = fake_chat_json  # type: ignore[method-assign]
    result = asyncio.run(client.verify("текст", []))
    assert result == []
    assert called["n"] == 0
