"""Смоук-тесты FastAPI с моками Ollama/OpenRouter.

Не запускают реальные модели — проверяют структуру эндпоинтов, аудит,
формат ответа /suggest и корректную обработку ошибок.
"""
import importlib
import io
import os
import re
import sys
from pathlib import Path

import httpx
import pytest


ROOT = Path(__file__).resolve().parent.parent


def _load_local_server(monkeypatch, tmp_path):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("AUDIT_DB", str(tmp_path / "audit.sqlite"))
    monkeypatch.setenv("RAG_ENABLED", "false")
    monkeypatch.setenv("MODEL_NAME", "qwen3:30b-a3b")

    # Чистый импорт: сбрасываем кеш, чтобы env подхватился
    for m in list(sys.modules):
        if m.startswith(("shared", "main")):
            sys.modules.pop(m, None)

    local_dir = ROOT / "server" / "local"
    sys.path.insert(0, str(local_dir))
    module = importlib.import_module("main")
    yield module
    sys.path.remove(str(local_dir))
    for m in list(sys.modules):
        if m.startswith(("shared", "main")):
            sys.modules.pop(m, None)


def _load_cloud_server(monkeypatch, tmp_path):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("AUDIT_DB", str(tmp_path / "audit.sqlite"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-key-123456")

    for m in list(sys.modules):
        if m.startswith(("shared", "main")):
            sys.modules.pop(m, None)

    cloud_dir = ROOT / "server" / "cloud"
    sys.path.insert(0, str(cloud_dir))
    module = importlib.import_module("main")
    yield module
    sys.path.remove(str(cloud_dir))
    for m in list(sys.modules):
        if m.startswith(("shared", "main")):
            sys.modules.pop(m, None)


@pytest.fixture
def local_module(monkeypatch, tmp_path):
    yield from _load_local_server(monkeypatch, tmp_path)


@pytest.fixture
def cloud_module(monkeypatch, tmp_path):
    yield from _load_cloud_server(monkeypatch, tmp_path)


def test_local_suggest_with_mocked_ollama(local_module, monkeypatch):
    from fastapi.testclient import TestClient

    async def fake_call_ollama(messages):
        return (
            "===CORRECTED===\n"
            "Согласно приказу №5.\n"
            "===CHANGES===\n"
            "1. «согласно приказа» → «согласно приказу» | предлог требует дательного падежа\n"
            "===END==="
        )

    monkeypatch.setattr(local_module, "call_ollama", fake_call_ollama)
    client = TestClient(local_module.app)
    files = {
        "text": ("t.txt", io.BytesIO("согласно приказа №5".encode("utf-8")), "text/plain"),
        "context": ("c.txt", io.BytesIO(b""), "text/plain"),
    }
    r = client.post("/suggest", files=files)
    assert r.status_code == 200
    body = r.text
    assert "===CORRECTED===" in body
    assert "===CHANGES===" in body
    assert "===END===" in body
    assert "дательного падежа" in body


def test_local_suggest_handles_bad_format(local_module, monkeypatch):
    from fastapi.testclient import TestClient

    async def fake_call_ollama(messages):
        return "Просто текст без маркеров"

    monkeypatch.setattr(local_module, "call_ollama", fake_call_ollama)
    client = TestClient(local_module.app)
    files = {
        "text": ("t.txt", io.BytesIO("проверка".encode("utf-8")), "text/plain"),
        "context": ("c.txt", io.BytesIO(b""), "text/plain"),
    }
    r = client.post("/suggest", files=files)
    assert r.status_code == 200
    assert "===CORRECTED===" in r.text
    assert "не распознан" in r.text


def test_local_metrics(local_module):
    from fastapi.testclient import TestClient
    client = TestClient(local_module.app)
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert data["server"] == "local"
    assert "audit" in data


def test_local_normalize_line_breaks_helper(local_module):
    """v2.2: local-сервер тоже нормализует переносы (Shift+Enter fix).

    \\r\\n → \\n\\n (paragraph), \\r → \\n\\n, \\n остаётся (мягкий перенос),
    U+2028 → \\n.  SYSTEM_PROMPT local-сервера не меняется — нормализация
    нужна только чтобы Ollama видела одинаковую конвенцию переносов
    независимо от платформы клиента.
    """
    fn = local_module._normalize_line_breaks
    assert fn("a\r\nb") == "a\n\nb"
    assert fn("a\rb") == "a\n\nb"
    assert fn("a\nb") == "a\nb"
    assert fn("a\u2028b") == "a\nb"
    assert fn("a\n\n\n\nb") == "a\n\nb"
    assert fn("") == ""


def test_local_soft_linebreak_preserved(local_module, monkeypatch):
    """Shift+Enter (одиночный \\n) НЕ должен превращаться в paragraph break
    при отправке в Ollama. До v2.2 расширение разбивало такой текст
    на разные абзацы — фиксим на стороне сервера-нормализатора."""
    from fastapi.testclient import TestClient

    captured: dict[str, str] = {}

    async def fake_call_ollama(messages):
        captured["user_msg"] = messages[-1]["content"]
        return (
            "===CORRECTED===\n"
            "Первая строка\nвторая строка.\n"
            "===CHANGES===\n"
            "1. Ошибок не найдено. Текст соответствует нормам.\n"
            "===END==="
        )

    monkeypatch.setattr(local_module, "call_ollama", fake_call_ollama)
    client = TestClient(local_module.app)
    raw = "Первая строка\nвторая строка.\r\nВторой абзац."
    files = {
        "text": ("t.txt", io.BytesIO(raw.encode("utf-8")), "text/plain"),
        "context": ("c.txt", io.BytesIO(b""), "text/plain"),
    }
    r = client.post("/suggest", files=files)
    assert r.status_code == 200
    sent = captured["user_msg"]
    assert "Первая строка\nвторая строка." in sent
    assert "\r" not in sent


def test_local_strip_thinking(local_module):
    """Проверка, что <think>…</think> обрезается из ответа Ollama."""
    raw = (
        "<think>Долго рассуждаю про падежи и согласование...</think>\n\n"
        "===CORRECTED===\nтекст\n===CHANGES===\n1. Ошибок нет.\n===END==="
    )
    out = local_module._strip_thinking(raw)
    assert "<think>" not in out
    assert "Долго рассуждаю" not in out
    assert out.startswith("===CORRECTED===")


def test_local_strip_thinking_without_tags(local_module):
    """Если модель пишет рассуждения без <think>, но дальше ===CORRECTED===,
    обрезаем всё до маркера."""
    raw = (
        "Пользователь хочет проверку текста. Подумаю над правилами...\n\n"
        "===CORRECTED===\nок\n===CHANGES===\n1. Ошибок нет.\n===END==="
    )
    out = local_module._strip_thinking(raw)
    assert out.startswith("===CORRECTED===")
    assert "Пользователь хочет" not in out


def test_local_drops_idempotent_changes(local_module):
    """Пункт вида «X → X» (before=after) фильтруется из ===CHANGES==="""
    raw = (
        "===CORRECTED===\n"
        "текст\n"
        "===CHANGES===\n"
        "1. «согласно распоряжению» — исправлено на «согласно распоряжению» (дательный падеж)\n"
        "2. «округе» → «округах» (множественное число)\n"
        "3. «отдел подготовил отчётность» — исправлено на «отдел подготовил отчётность» (без изменений)\n"
        "===END==="
    )
    out = local_module._drop_idempotent_changes(raw)
    # Первый и третий пункт должны исчезнуть; остался только содержательный
    assert "«округах»" in out
    assert "«согласно распоряжению»" not in out
    assert "отдел подготовил отчётность" not in out
    # Рамки сохранены
    assert "===CORRECTED===" in out
    assert "===CHANGES===" in out
    assert "===END===" in out


def test_local_drops_idempotent_changes_empty_fallback(local_module):
    """Если после фильтрации пунктов не осталось — подставляем заглушку."""
    raw = (
        "===CORRECTED===\n"
        "текст\n"
        "===CHANGES===\n"
        "1. «фраза» → «фраза»\n"
        "2. «другая» — исправлено на «другая»\n"
        "===END==="
    )
    out = local_module._drop_idempotent_changes(raw)
    assert "Ошибок не найдено" in out
    assert "===END===" in out


def test_local_case_correction_is_preserved(local_module):
    """Правка только регистра (Приказа → приказа) — валидная орфографическая,
    НЕ должна считаться идемпотентной и НЕ должна удаляться из ===CHANGES===."""
    raw = (
        "===CORRECTED===\n"
        "на основании приказа...\n"
        "===CHANGES===\n"
        "1. «Приказа» → «приказа» (слово не является именем собственным)\n"
        "2. «округе» → «округе» (ложная правка — дублирует текст)\n"
        "===END==="
    )
    out = local_module._drop_idempotent_changes(raw)
    # Правка регистра сохранена
    assert "«Приказа» → «приказа»" in out
    # Идемпотентная отфильтрована
    assert "«округе» → «округе»" not in out
    # Заглушка НЕ подставлена (остался содержательный пункт)
    assert "Ошибок не найдено" not in out


def test_local_drops_changes_with_ellipsis_in_quotes(local_module):
    """Пункты с многоточием («...» или «…») в цитатах неприменимы клиентом
    (InStr не найдёт сокращённую цитату в выделении). Сервер их отбрасывает,
    чтобы пользователь не видел «не удалось применить» на каждом запросе.

    Видели на yandex-corrector / Yandex-template моделях — стилистически
    сокращают длинные цитаты. T-lite и qwen2.5 этим почти не страдают."""
    raw = (
        "===CORRECTED===\n"
        "текст\n"
        "===CHANGES===\n"
        "1. «по обеспечению...» → «по обеспечению,...» | пропущенная запятая\n"
        "2. «короткая фраза» → «короткая, фраза» | запятая\n"
        "3. «другой пример…» → «другой, пример…» | unicode-многоточие\n"
        "===END==="
    )
    out = local_module._drop_idempotent_changes(raw)
    # Пункты 1 и 3 (многоточие в цитате) отфильтрованы
    assert "по обеспечению" not in out
    assert "другой пример" not in out
    # Пункт 2 (без многоточия) сохранён
    assert "«короткая фраза» → «короткая, фраза»" in out
    # Заглушка НЕ подставлена (остался содержательный пункт)
    assert "Ошибок не найдено" not in out


def test_v182_drops_idempotent_with_nested_quotes(local_module):
    """v1.8.2 регрессия: пункт «адм…здания «ЦСН ВО»» → «адм…здания «ЦСН ВО»»
    с ВЛОЖЕННЫМИ «...» в before/after. До v1.8.2 _CHANGE_PAIR_RE захватывал
    внутреннюю пару «ЦСН ВО» как before и срез ` → ` как after — before != after,
    идемпотентный пункт пропускался в выдачу. После v1.8.2 robust-парсер
    жадно матчит внешние «...» и обнаруживает совпадение.
    """
    raw = (
        "===CORRECTED===\n"
        "текст\n"
        "===CHANGES===\n"
        "1. «кварталах» → «квартале» | согласование\n"
        "2. «административного здания «ЦСН ВО»» → «административного здания «ЦСН ВО»» | пунктуация\n"
        "===END==="
    )
    out = local_module._drop_idempotent_changes(raw)
    # Идемпотентный пункт с вложенными кавычками удалён
    assert "не требуется" not in out
    assert "пунктуация" not in out
    # Содержательный пункт сохранён
    assert "«кварталах» → «квартале»" in out
    # Заглушка не подставлена
    assert "Ошибок не найдено" not in out


def test_v182_parse_change_pair_robust(local_module):
    """v1.8.2: robust-парсер корректно извлекает (before, after) при
    вложенных кавычках и игнорирует ` | explanation`."""
    fn = local_module._parse_change_pair_robust
    # Простой случай
    assert fn("1. «кварталах» → «квартале» | согласование") == ("кварталах", "квартале")
    # Вложенные кавычки в before и after
    assert fn(
        '3. «административного здания «ЦСН ВО»» → «административного здания «ЦСН ВО»» | пунктуация'
    ) == ("административного здания «ЦСН ВО»", "административного здания «ЦСН ВО»")
    # Без ` | ` (пояснения нет)
    assert fn("«старое» → «новое»") == ("старое", "новое")
    # Не CHANGES-строка — None
    assert fn("просто текст без стрелки") is None
    assert fn("") is None


def test_local_drops_changes_with_hallucinated_before(local_module):
    """Пункты, чьё «было» не является подстрокой raw_text — галлюцинации
    модели. Сервер их дропает на финальном этапе."""
    raw_text = (
        "Главным управлением собственной безопасности проверяется "
        "информация о противоправных действиях."
    )
    response = (
        "===CORRECTED===\n"
        "Главным управлением собственной безопасности проверяется информация о противоправных действиях.\n"
        "===CHANGES===\n"
        "1. «безопасностей» → «безопасности» | падеж\n"
        "2. «противоправных» → «правонарушительных» | синонимы\n"
        "3. «информация о» → «информация про» | предлог\n"
        "===END==="
    )
    out = local_module._drop_changes_not_in_text(response, raw_text)
    # Пункт 1: «безопасностей» нет в raw_text — выкидывается
    assert "безопасностей" not in out
    # Пункт 2: «противоправных» есть в raw_text — остаётся (даже если правка спорная)
    assert "противоправных" in out
    # Пункт 3: «информация о» есть в raw_text — остаётся
    assert "информация о" in out
    # Заглушка НЕ подставлена (есть содержательные пункты)
    assert "Ошибок не найдено" not in out


def test_local_drops_all_hallucinated_uses_fallback(local_module):
    """Если ВСЕ пункты галлюцинированные — подставляется заглушка."""
    raw_text = "Простой короткий текст без ошибок."
    response = (
        "===CORRECTED===\n"
        "Простой короткий текст без ошибок.\n"
        "===CHANGES===\n"
        "1. «несуществующее слово» → «другое» | замена\n"
        "2. «ещё одна выдумка» → «правильно» | замена\n"
        "===END==="
    )
    out = local_module._drop_changes_not_in_text(response, raw_text)
    assert "Ошибок не найдено" in out
    assert "===END===" in out


def test_local_passes_through_when_raw_text_empty(local_module):
    """Если raw_text пуст — фильтр выключен (не ломаем тесты с моком)."""
    response = (
        "===CORRECTED===\n"
        "текст\n"
        "===CHANGES===\n"
        "1. «X» → «Y» | замена\n"
        "===END==="
    )
    out = local_module._drop_changes_not_in_text(response, "")
    assert "«X» → «Y»" in out


# ─── v1.7.3: _renumber_changes ─────────────────────────────────────────


def test_renumber_changes_fills_gaps(local_module):
    """v1.7.3 prod-кейс: фильтр дропнул пункт «1.», осталось «2. ... 3. ...»
    в LibreOffice-расширении пользователь видит начало с 2-го пункта.
    После _renumber_changes нумерация идёт подряд: 1, 2."""
    raw = (
        "===CORRECTED===\n"
        "текст\n"
        "===CHANGES===\n"
        "2. «работ,» → «работы,» | согласование\n"
        "3. «не предусмотренной» → «не предусмотренных» | агремент\n"
        "===END==="
    )
    out = local_module._renumber_changes(raw)
    assert "1. «работ,» → «работы,»" in out
    assert "2. «не предусмотренной» → «не предусмотренных»" in out
    # Старые номера 2/3 не должны остаться
    assert "2. «работ,»" not in out
    assert "3. «не предусмотренной»" not in out


def test_renumber_changes_already_sequential(local_module):
    """v1.7.3: если нумерация уже сплошная — оставить как есть."""
    raw = (
        "===CORRECTED===\n"
        "текст\n"
        "===CHANGES===\n"
        "1. «X» → «Y»\n"
        "2. «A» → «B»\n"
        "3. «C» → «D»\n"
        "===END==="
    )
    out = local_module._renumber_changes(raw)
    assert "1. «X» → «Y»" in out
    assert "2. «A» → «B»" in out
    assert "3. «C» → «D»" in out


def test_renumber_changes_empty_lines_preserved(local_module):
    """v1.7.3: пустые строки между пунктами не считаются и не нумеруются."""
    raw = (
        "===CORRECTED===\n"
        "текст\n"
        "===CHANGES===\n"
        "\n"
        "5. «X» → «Y»\n"
        "\n"
        "10. «A» → «B»\n"
        "===END==="
    )
    out = local_module._renumber_changes(raw)
    assert "1. «X» → «Y»" in out
    assert "2. «A» → «B»" in out
    assert "5. «X»" not in out
    assert "10. «A»" not in out


def test_renumber_changes_no_changes_block(local_module):
    """Если нет ===CHANGES===/===END=== — текст возвращается без изменений."""
    raw = "просто текст без блоков"
    assert local_module._renumber_changes(raw) == raw


def test_renumber_changes_oshibok_ne_naydeno_unaffected(local_module):
    """v1.7.3: «Ошибок не найдено» — единственный пункт, всё ок."""
    raw = (
        "===CORRECTED===\n"
        "текст\n"
        "===CHANGES===\n"
        "1. Ошибок не найдено. Текст соответствует нормам.\n"
        "===END==="
    )
    out = local_module._renumber_changes(raw)
    assert "1. Ошибок не найдено" in out


# ─── v1.6.8: фильтр стилистических ё-замен ──────────────────────


def test_is_eyo_only_substitution_basic(local_module):
    """Чисто ё↔е подмены распознаются как стилистика."""
    f = local_module._is_eyo_only_substitution
    assert f("проведенными", "проведёнными") is True
    assert f("проведёнными", "проведенными") is True
    assert f("повлекших", "повлёкших") is True
    assert f("Тёркин", "Теркин") is True  # с большой буквы тоже
    # Реальные правки НЕ распознаются как стилистика
    assert f("стоимостей", "стоимости") is False  # число
    assert f("выполненной", "выполненных") is False  # причастие
    assert f("Подразделения", "Подразделению") is False  # падеж
    # Пустые строки и совпадение
    assert f("", "проведёнными") is False
    assert f("проведенными", "") is False
    assert f("проведенными", "проведенными") is False


def test_drop_eyo_substitutions_filters_changes_and_undoes_corrected(local_module):
    """Пункт «проведенными → проведёнными» — стилистика. Сервер должен:
    1) убрать пункт из ===CHANGES===,
    2) откатить замену в ===CORRECTED=== (вернуть «проведенными»).
    """
    raw_text = (
        "Проведенными мероприятиями установлены факты, повлекших риски."
    )
    response = (
        "===CORRECTED===\n"
        "Проведёнными мероприятиями установлены факты, повлёкших риски.\n"
        "===CHANGES===\n"
        "1. «Проведенными» → «Проведёнными» | е/ё в причастии\n"
        "2. «повлекших» → «повлёкших» | е/ё в причастии\n"
        "===END==="
    )
    out = local_module._drop_eyo_substitutions(response, raw_text)
    # CORRECTED откатан к исходному написанию
    assert "Проведенными" in out
    assert "Проведёнными" not in out
    assert "повлекших" in out
    assert "повлёкших" not in out
    # Оба пункта-стилистики удалены
    assert "1. «Проведенными»" not in out
    assert "2. «повлекших»" not in out
    # Так как все пункты были стилистикой — подставлена заглушка
    assert "Ошибок не найдено" in out


def test_drop_eyo_substitutions_keeps_real_changes(local_module):
    """Реальные правки (число, падеж, причастие) НЕ должны фильтроваться."""
    raw_text = "стоимостей выполненной работ путём применения"
    response = (
        "===CORRECTED===\n"
        "стоимости выполненных работ путём применения\n"
        "===CHANGES===\n"
        "1. «стоимостей» → «стоимости» | число существительного\n"
        "2. «выполненной» → «выполненных» | согласование причастия\n"
        "===END==="
    )
    out = local_module._drop_eyo_substitutions(response, raw_text)
    # Обе правки сохранены — это не ё-замены
    assert "«стоимостей» → «стоимости»" in out
    assert "«выполненной» → «выполненных»" in out
    # CORRECTED не тронут
    assert "стоимости выполненных работ" in out


def test_drop_eyo_substitutions_mixed_eyo_and_real(local_module):
    """Смешанный случай: одна ё-замена + одна реальная правка.
    Стилистическая дропается, реальная остаётся."""
    raw_text = "Проведенными мероприятиями выявлены стоимостей выполненной работ."
    response = (
        "===CORRECTED===\n"
        "Проведёнными мероприятиями выявлены стоимости выполненных работ.\n"
        "===CHANGES===\n"
        "1. «Проведенными» → «Проведёнными» | е/ё\n"
        "2. «стоимостей выполненной» → «стоимости выполненных» | согласование\n"
        "===END==="
    )
    out = local_module._drop_eyo_substitutions(response, raw_text)
    # ё-замена удалена и откатана в CORRECTED
    assert "Проведенными" in out
    assert "Проведёнными" not in out
    assert "1. «Проведенными»" not in out
    # Реальная правка осталась
    assert "«стоимостей выполненной» → «стоимости выполненных»" in out
    # CORRECTED содержит и откат ё, и реальную правку
    assert "Проведенными мероприятиями выявлены стоимости выполненных работ" in out


def test_drop_eyo_substitutions_no_op_when_no_eyo(local_module):
    """Если ё-замен нет — функция возвращает текст без изменений (короткий путь)."""
    raw_text = "стоимостей выполненной работ"
    response = (
        "===CORRECTED===\n"
        "стоимости выполненных работ\n"
        "===CHANGES===\n"
        "1. «стоимостей» → «стоимости» | число\n"
        "===END==="
    )
    out = local_module._drop_eyo_substitutions(response, raw_text)
    assert out == response  # идентично — никаких изменений


def test_drop_eyo_substitutions_safe_on_empty_raw(local_module):
    """На пустом raw_text функция не падает и ничего не меняет."""
    response = (
        "===CORRECTED===\n"
        "текст\n"
        "===CHANGES===\n"
        "1. «проведенными» → «проведёнными» | е/ё\n"
        "===END==="
    )
    out = local_module._drop_eyo_substitutions(response, "")
    assert out == response


def test_drop_eyo_substitutions_safe_on_malformed_response(local_module):
    """Если в ответе нет CORRECTED/CHANGES/END — функция не падает."""
    f = local_module._drop_eyo_substitutions
    assert f("просто текст", "raw") == "просто текст"
    assert f("===CORRECTED=== без секций", "raw") == "===CORRECTED=== без секций"


# ─── v1.6.9: посимвольный char-level откат ё→е в CORRECTED ──────


def test_undo_eyo_in_text_simple(local_module):
    """Простой случай: в corrected ё там, где в raw — е. Откат восстанавливает."""
    raw = "проведенными мероприятиями повлекших риски"
    corrected = "проведёнными мероприятиями повлёкших риски"
    out = local_module._undo_eyo_in_text(corrected, raw)
    assert out == "проведенными мероприятиями повлекших риски"


def test_undo_eyo_in_text_compound_bypass(local_module):
    """Реальный кейс v1.6.8 prod (run1, 5 мая 2026, КС-2):
    модель упаковала ё-подмену + падежную в одну compound-цитату:
    «повлекших риски ... Подразделения» → «повлёкших риски ... Подразделению».
    Line-level фильтр такое не дропает (не чистая ё-замена), но
    char-level откат должен восстановить «повлекших», оставив
    падежную правку «Подразделению» (это НЕ ё, длина сегмента не та)."""
    raw = "нарушений, повлекших риски причинения ущерба Подразделения в размере"
    corrected = "нарушений, повлёкших риски причинения ущерба Подразделению в размере"
    out = local_module._undo_eyo_in_text(corrected, raw)
    # ё-подмена откатана
    assert "повлекших" in out
    assert "повлёкших" not in out
    # Падежная (НЕ ё) — остаётся
    assert "Подразделению" in out


def test_undo_eyo_in_text_keeps_legitimate_e(local_module):
    """Реальная буква ё в raw_text сохраняется в corrected.
    Например, имя «Тёркин» — в raw оно с ё, в corrected тоже должно
    остаться с ё."""
    raw = "В произведении упомянут Тёркин."
    corrected = "В произведении упомянут Тёркин."
    out = local_module._undo_eyo_in_text(corrected, raw)
    assert out == corrected
    # И никакой откат не нужен — счётчик undone == 0


def test_undo_eyo_in_text_keeps_real_corrections(local_module):
    """Реальные правки (НЕ ё) в corrected не трогаем."""
    raw = "стоимостей выполненной работ"
    corrected = "стоимости выполненных работ"
    out = local_module._undo_eyo_in_text(corrected, raw)
    assert out == corrected  # без изменений: нет ё в corrected


def test_undo_eyo_in_text_uppercase(local_module):
    """Большая Ё→Е тоже откатывается."""
    raw = "Елена Проведенными мероприятиями"
    corrected = "Ёлена Проведёнными мероприятиями"
    out = local_module._undo_eyo_in_text(corrected, raw)
    assert out == "Елена Проведенными мероприятиями"


def test_undo_eyo_in_text_no_eyo_short_circuit(local_module):
    """Если в corrected нет ни ё, ни Ё — функция возвращает ту же строку."""
    raw = "что-то"
    corrected = "что-то другое"
    out = local_module._undo_eyo_in_text(corrected, raw)
    assert out == corrected


def test_undo_eyo_in_text_empty_raw(local_module):
    """На пустом raw_text функция не падает и не меняет corrected."""
    out = local_module._undo_eyo_in_text("проведёнными", "")
    assert out == "проведёнными"


def test_undo_eyo_in_corrected_block_full_response(local_module):
    """Полный pipeline: ===CORRECTED=== содержит ё-подмену в compound,
    ===CHANGES=== не трогаем — это работа _drop_eyo_substitutions."""
    raw_text = (
        "нарушений, повлекших риски причинения ущерба Подразделения в размере"
    )
    response = (
        "===CORRECTED===\n"
        "нарушений, повлёкших риски причинения ущерба Подразделению в размере\n"
        "===CHANGES===\n"
        "1. «повлекших ... Подразделения» → «повлёкших ... Подразделению» | compound\n"
        "===END==="
    )
    out = local_module._undo_eyo_in_corrected_block(response, raw_text)
    # CORRECTED-блок откатан в части ё; падежная правка сохранена.
    corrected_block = out.split("===CORRECTED===", 1)[1].split("===CHANGES===", 1)[0]
    assert "повлекших" in corrected_block
    assert "повлёкших" not in corrected_block
    assert "Подразделению" in corrected_block
    # CHANGES не тронут — пункт целиком сохранён, включая ё-варианты в цитате.
    assert "1. «повлекших ... Подразделения» → «повлёкших ... Подразделению»" in out


def test_undo_eyo_in_corrected_block_safe_on_malformed(local_module):
    """На некорректном формате не падает."""
    f = local_module._undo_eyo_in_corrected_block
    assert f("просто текст", "raw") == "просто текст"
    assert f("===CORRECTED=== без CHANGES", "raw") == "===CORRECTED=== без CHANGES"


# ─── v1.6.10: warmup с реалистичным num_ctx + temperature=0 ──────


class _CapturingFakeHttpxClient:
    """Минимальный мок httpx.AsyncClient для тестов v1.6.10. Захватывает
    payload последнего POST и возвращает фиксированный успешный ответ.
    Используется как drop-in замена `httpx.AsyncClient` в тестах warmup
    и call_ollama, чтобы можно было проверить структуру JSON-payload без
    реального обращения к Ollama."""

    captured: dict = {}

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, url, json=None, **kwargs):
        type(self).captured = {"url": url, "json": json}

        class _Resp:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "message": {
                        "content": (
                            "===CORRECTED===\nок\n"
                            "===CHANGES===\n1. Ошибок нет.\n===END==="
                        )
                    }
                }

        return _Resp()


def test_local_warmup_uses_full_num_ctx(local_module, monkeypatch):
    """v1.6.10: прогрев должен использовать тот же `num_ctx`, что и реальные
    запросы (`OLLAMA_NUM_CTX`). Иначе Ollama при первом реальном запросе
    переаллоцирует kv-cache (раньше: warmup `num_ctx=512` → реальный
    `num_ctx=2048-4096` = +15-25 с cold-старта).

    Регрессионный тест на v1.6.8 prod-инцидент (5 мая 2026, КС-2):
    cold=100с, warm=79с — 21с разницы списать только на kv-cache resize."""
    import asyncio

    _CapturingFakeHttpxClient.captured = {}
    monkeypatch.setattr(local_module.httpx, "AsyncClient", _CapturingFakeHttpxClient)

    asyncio.run(local_module._warmup_ollama())

    payload = _CapturingFakeHttpxClient.captured.get("json", {})
    assert _CapturingFakeHttpxClient.captured.get("url", "").endswith("/api/chat")
    options = payload.get("options", {})
    assert options.get("num_ctx") == local_module.OLLAMA_NUM_CTX, (
        f"v1.6.10: warmup должен слать num_ctx=OLLAMA_NUM_CTX="
        f"{local_module.OLLAMA_NUM_CTX}, получено {options.get('num_ctx')}"
    )


def test_local_warmup_passes_temperature(local_module, monkeypatch):
    """v1.6.10: прогрев передаёт `temperature` в Ollama-options. Без этого
    Ollama использует свой дефолт (0.7-0.8), и при первом реальном запросе
    с temperature=0 пересобирает sampling-state — небольшое, но добавляет
    к cold-start latency."""
    import asyncio

    _CapturingFakeHttpxClient.captured = {}
    monkeypatch.setattr(local_module.httpx, "AsyncClient", _CapturingFakeHttpxClient)

    asyncio.run(local_module._warmup_ollama())

    options = _CapturingFakeHttpxClient.captured.get("json", {}).get("options", {})
    assert "temperature" in options, "Прогрев должен явно передавать temperature"
    assert options["temperature"] == local_module.OLLAMA_TEMPERATURE, (
        f"v1.6.10: warmup-temperature должен совпадать с OLLAMA_TEMPERATURE="
        f"{local_module.OLLAMA_TEMPERATURE}"
    )


def test_local_warmup_uses_realistic_prompt(local_module, monkeypatch):
    """v1.6.10: прогрев слать промпт реалистичной длины (>=500 chars),
    чтобы Ollama скомпилировала attention pattern и токенизатор именно
    под русский текст. До v1.6.10 был промпт «ok» (4 chars) — это
    прогревало weights, но не decode-loop."""
    import asyncio

    _CapturingFakeHttpxClient.captured = {}
    monkeypatch.setattr(local_module.httpx, "AsyncClient", _CapturingFakeHttpxClient)

    asyncio.run(local_module._warmup_ollama())

    messages = _CapturingFakeHttpxClient.captured.get("json", {}).get("messages", [])
    assert messages, "Прогрев должен слать messages"
    user_msg = messages[-1]["content"]
    assert len(user_msg) >= 500, (
        f"v1.6.10: warmup-промпт слишком короткий ({len(user_msg)} chars), "
        f"должен быть ≥500 для прогрева decode-пути"
    )


def test_local_warmup_skipped_when_disabled(local_module, monkeypatch):
    """OLLAMA_WARMUP=false: прогрев не должен делать HTTP-запросов."""
    import asyncio

    _CapturingFakeHttpxClient.captured = {}
    monkeypatch.setattr(local_module, "OLLAMA_WARMUP", False)
    monkeypatch.setattr(local_module.httpx, "AsyncClient", _CapturingFakeHttpxClient)

    asyncio.run(local_module._warmup_ollama())

    assert _CapturingFakeHttpxClient.captured == {}, (
        "При OLLAMA_WARMUP=false прогрев не должен открывать httpx-сессию"
    )


def test_local_call_ollama_default_temperature_zero(local_module, monkeypatch):
    """v1.6.10: дефолтная temperature=0 для greedy/детерминированной
    генерации. До v1.6.10 хардкодилось 0.1 — это давало малую, но
    воспроизводимую вариативность ответа: одинаковый текст мог получить
    разные CHANGES (в v1.6.8 ablation: run1=1 пункт detailed,
    run2=3 пункта diff-reconstruction). Это маскировало регрессии в QA."""
    import asyncio

    _CapturingFakeHttpxClient.captured = {}
    monkeypatch.setattr(local_module.httpx, "AsyncClient", _CapturingFakeHttpxClient)

    asyncio.run(
        local_module.call_ollama([{"role": "user", "content": "тестовый запрос"}])
    )

    options = _CapturingFakeHttpxClient.captured.get("json", {}).get("options", {})
    assert options.get("temperature") == 0.0, (
        f"v1.6.10: дефолтная temperature должна быть 0 (greedy), "
        f"получено {options.get('temperature')}"
    )


# ─── v1.7: pymorphy3 фильтр падежных «улучшений» ──────────────────


def _has_pymorphy3() -> bool:
    try:
        import pymorphy3  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_pymorphy3(), reason="pymorphy3 не установлен")
def test_local_drop_morph_case_substitution_подразделения(local_module):
    """v1.7 главный prod-кейс: модель упаковала compound-цитату с
    ё-заменой и падежной подменой. После v1.6.9 ё откатывается, но
    падежная подмена «Подразделения → Подразделению» остаётся в CORRECTED.
    Морф-фильтр должен дропнуть и откатить.

    Важно: тест эмулирует случай, когда модель отдала ОДИНОЧНУЮ цитату
    «Подразделения» → «Подразделению» (после того, как другие фильтры
    разобрали compound). Морф-фильтр работает на одиночных словах."""
    f = local_module._drop_morph_case_substitutions
    if local_module._morph_filter is None or not local_module._morph_filter.available:
        pytest.skip("MorphFilter не активен в среде теста")
    raw = (
        "повлекших риски причинения ущерба Подразделения в размере более "
        "2 млн рублей"
    )
    response = (
        "===CORRECTED===\n"
        "повлекших риски причинения ущерба Подразделению в размере более "
        "2 млн рублей\n"
        "===CHANGES===\n"
        "1. «Подразделения» → «Подразделению» | согласование падежа\n"
        "===END==="
    )
    out = f(response, raw)
    # Пункт CHANGES должен быть выкинут (если он был единственным —
    # заменён на «Ошибок не найдено»).
    assert "«Подразделения» → «Подразделению»" not in out
    # CORRECTED должен быть откатан до исходной формы Подразделения.
    assert "Подразделения" in out
    assert "Подразделению" not in out


@pytest.mark.skipif(not _has_pymorphy3(), reason="pymorphy3 не установлен")
def test_local_drop_morph_keeps_согласно_приказа(local_module):
    """Реальная ошибка управления: «согласно приказа» (gent) → «согласно
    приказу» (datv). «согласно» — case-governing предлог, фильтр должен
    ОСТАВИТЬ правку (НЕ дропать)."""
    f = local_module._drop_morph_case_substitutions
    if local_module._morph_filter is None or not local_module._morph_filter.available:
        pytest.skip("MorphFilter не активен в среде теста")
    raw = "согласно приказа №5 от 12.05.2026 проведено мероприятие"
    response = (
        "===CORRECTED===\n"
        "согласно приказу №5 от 12.05.2026 проведено мероприятие\n"
        "===CHANGES===\n"
        "1. «приказа» → «приказу» | предлог согласно требует дательного падежа\n"
        "===END==="
    )
    out = f(response, raw)
    # Пункт CHANGES должен сохраниться.
    assert "«приказа» → «приказу»" in out
    # CORRECTED должен остаться в исправленной форме (НЕ откатан).
    assert "согласно приказу" in out


@pytest.mark.skipif(not _has_pymorphy3(), reason="pymorphy3 не установлен")
def test_local_drop_morph_keeps_number_change(local_module):
    """Изменение числа — agreement fix, должен остаться:
    «выполненной» (sing) → «выполненных» (plur)."""
    f = local_module._drop_morph_case_substitutions
    if local_module._morph_filter is None or not local_module._morph_filter.available:
        pytest.skip("MorphFilter не активен в среде теста")
    raw = "стоимостей выполненной работ путём завышения расценок"
    response = (
        "===CORRECTED===\n"
        "стоимостей выполненных работ путём завышения расценок\n"
        "===CHANGES===\n"
        "1. «выполненной» → «выполненных» | согласование числа с однородными\n"
        "===END==="
    )
    out = f(response, raw)
    assert "«выполненной» → «выполненных»" in out
    assert "выполненных работ" in out


@pytest.mark.skipif(not _has_pymorphy3(), reason="pymorphy3 не установлен")
def test_local_drop_morph_keeps_lexical_change(local_module):
    """Лексическая замена (разные леммы) — должен остаться."""
    f = local_module._drop_morph_case_substitutions
    if local_module._morph_filter is None or not local_module._morph_filter.available:
        pytest.skip("MorphFilter не активен в среде теста")
    raw = "комиссия должна принимать решение оперативно"
    response = (
        "===CORRECTED===\n"
        "комиссия должна принять решение оперативно\n"
        "===CHANGES===\n"
        "1. «принимать» → «принять» | вид глагола\n"
        "===END==="
    )
    out = f(response, raw)
    assert "«принимать» → «принять»" in out


@pytest.mark.skipif(not _has_pymorphy3(), reason="pymorphy3 не установлен")
def test_local_drop_morph_compound_main_prod_case(local_module):
    """v1.7.1 главный prod-кейс (КС-2 6 мая 2026): модель отдала
    compound-цитату «повлекших риски причинения ущерба Подразделения»
    → «повлёкших риски причинения ущерба Подразделению». Внутри
    ё-различие («повлекших» → «повлёкших») и галлюцинированная
    падежная подмена («Подразделения» → «Подразделению»). Single-word
    путь морф-фильтра проигнорировал бы (есть пробелы); compound-путь
    должен:
      1. Откатить Подразделению → Подразделения в CORRECTED.
      2. Дропнуть весь пункт CHANGES (все нетривиальные различия —
         галлюцинации, реальной правки нет).
    """
    f = local_module._drop_morph_case_substitutions
    if local_module._morph_filter is None or not local_module._morph_filter.available:
        pytest.skip("MorphFilter не активен в среде теста")
    raw = (
        "ряд значительных нарушений, повлекших риски причинения "
        "ущерба Подразделения в размере более 2 млн рублей."
    )
    response = (
        "===CORRECTED===\n"
        "ряд значительных нарушений, повлекших риски причинения "
        "ущерба Подразделению в размере более 2 млн рублей.\n"
        "===CHANGES===\n"
        "1. «повлекших риски причинения ущерба Подразделения» "
        "→ «повлёкших риски причинения ущерба Подразделению» "
        "| согласование причастий и существительных\n"
        "===END==="
    )
    out = f(response, raw)
    # Compound-пункт должен быть выкинут.
    assert "повлёкших риски причинения ущерба Подразделению" not in out
    # CORRECTED должен быть откатан: Подразделению → Подразделения.
    assert "ущерба Подразделения" in out
    assert "ущерба Подразделению" not in out


@pytest.mark.skipif(not _has_pymorphy3(), reason="pymorphy3 не установлен")
def test_local_drop_morph_compound_mixed_keeps_item(local_module):
    """v1.7.1: compound с реальной правкой числа («выполненной» →
    «выполненных») И галлюцинацией («Подразделения» → «Подразделению»)
    — пункт CHANGES должен остаться (там реальная правка), но
    галлюцинированное слово должно быть откатано в CORRECTED."""
    f = local_module._drop_morph_case_substitutions
    if local_module._morph_filter is None or not local_module._morph_filter.available:
        pytest.skip("MorphFilter не активен в среде теста")
    raw = "выполненной работ ущерба Подразделения"
    response = (
        "===CORRECTED===\n"
        "выполненных работ ущерба Подразделению\n"
        "===CHANGES===\n"
        "1. «выполненной работ ущерба Подразделения» "
        "→ «выполненных работ ущерба Подразделению» "
        "| согласование причастия с дополнением\n"
        "===END==="
    )
    out = f(response, raw)
    corrected = out.split("===CORRECTED===", 1)[1].split("===CHANGES===", 1)[0]
    changes = out.split("===CHANGES===", 1)[1].split("===END===", 1)[0]
    # CHANGES line должна остаться (там есть реальная правка числа).
    assert "выполненных работ ущерба Подразделению" in changes
    # Но Подразделению должно быть откатано в CORRECTED, а реальная
    # правка числа сохранена.
    assert "выполненных работ" in corrected
    assert "ущерба Подразделения" in corrected
    assert "Подразделению" not in corrected


@pytest.mark.skipif(not _has_pymorphy3(), reason="pymorphy3 не установлен")
def test_local_drop_morph_compound_keeps_governing_prep(local_module):
    """v1.7.1: compound, где падежная подмена прикрыта case-governing
    предлогом («согласно приказа» → «согласно приказу»), не должен
    дропаться даже внутри компаунда."""
    f = local_module._drop_morph_case_substitutions
    if local_module._morph_filter is None or not local_module._morph_filter.available:
        pytest.skip("MorphFilter не активен в среде теста")
    raw = "согласно приказа от 12.05.2026 проведено мероприятие"
    response = (
        "===CORRECTED===\n"
        "согласно приказу от 12.05.2026 проведено мероприятие\n"
        "===CHANGES===\n"
        "1. «согласно приказа от» → «согласно приказу от» "
        "| предлог согласно требует дательного падежа\n"
        "===END==="
    )
    out = f(response, raw)
    corrected = out.split("===CORRECTED===", 1)[1].split("===CHANGES===", 1)[0]
    changes = out.split("===CHANGES===", 1)[1].split("===END===", 1)[0]
    # CHANGES line должна сохраниться (реальная ошибка управления).
    assert "согласно приказу от" in changes
    # CORRECTED НЕ откатан: остаётся в исправленной форме (приказу).
    assert "согласно приказу от" in corrected
    assert "согласно приказа от" not in corrected


@pytest.mark.skipif(not _has_pymorphy3(), reason="pymorphy3 не установлен")
def test_local_drop_morph_safe_on_malformed(local_module):
    """На некорректном формате не падает, возвращает вход без изменений."""
    f = local_module._drop_morph_case_substitutions
    assert f("просто текст", "raw") == "просто текст"
    assert f("===CORRECTED=== без CHANGES", "raw") == "===CORRECTED=== без CHANGES"


@pytest.mark.skipif(not _has_pymorphy3(), reason="pymorphy3 не установлен")
def test_local_drop_morph_safe_on_empty_raw(local_module):
    """Пустой raw_text — фильтр пропускает (нет контекста для проверки)."""
    f = local_module._drop_morph_case_substitutions
    response = (
        "===CORRECTED===\nок\n"
        "===CHANGES===\n1. «Подразделения» → «Подразделению» | падеж\n"
        "===END==="
    )
    # При пустом raw_text фильтр должен выйти на гарде (return text).
    assert f(response, "") == response


def test_local_drop_morph_disabled_when_filter_unavailable(local_module, monkeypatch):
    """Если _morph_filter is None или not available — фильтр no-op,
    пайплайн возвращает input без изменений."""
    f = local_module._drop_morph_case_substitutions
    monkeypatch.setattr(local_module, "_morph_filter", None)
    raw = "повлекших риски причинения ущерба Подразделения в размере"
    response = (
        "===CORRECTED===\n"
        "повлекших риски причинения ущерба Подразделению в размере\n"
        "===CHANGES===\n"
        "1. «Подразделения» → «Подразделению» | согласование падежа\n"
        "===END==="
    )
    # _morph_filter=None → no-op, текст возвращается as-is.
    assert f(response, raw) == response


def test_local_call_ollama_respects_temperature_override(local_module, monkeypatch):
    """OLLAMA_TEMPERATURE можно переопределить (например, для exploration
    в исследовательских прогонах). Проверяем, что значение пробрасывается
    в Ollama-options без модификации."""
    import asyncio

    _CapturingFakeHttpxClient.captured = {}
    monkeypatch.setattr(local_module, "OLLAMA_TEMPERATURE", 0.35)
    monkeypatch.setattr(local_module.httpx, "AsyncClient", _CapturingFakeHttpxClient)

    asyncio.run(
        local_module.call_ollama([{"role": "user", "content": "тестовый запрос"}])
    )

    options = _CapturingFakeHttpxClient.captured.get("json", {}).get("options", {})
    assert options.get("temperature") == 0.35


def test_local_rebuild_changes_from_diff_punctuation(local_module):
    """Если модель добавила запятые в CORRECTED, но не отрапортовала —
    сервер должен сгенерировать пункты CHANGES из diff. Реальный кейс
    с yandex-corrector на Росгвардии."""
    raw = "в ходе выполнения задач по обеспечению собственной безопасности"
    corrected = "в ходе выполнения задач, по обеспечению собственной безопасности"
    entries = local_module._rebuild_changes_from_diff(raw, corrected)
    assert len(entries) == 1
    # «было» содержит исходник с контекстом ±1 слово вокруг запятой
    assert "задач" in entries[0] and "по" in entries[0]
    # «стало» содержит запятую
    assert "задач," in entries[0]
    # «было» — substring исходника (инвариант для клиента)
    before = entries[0].split("»")[0].lstrip("«")
    assert before in raw


def test_local_rebuild_changes_skips_when_equal(local_module):
    """Если raw_text и corrected совпадают — entries пуст."""
    raw = "Текст без правок."
    entries = local_module._rebuild_changes_from_diff(raw, raw)
    assert entries == []


def test_local_rebuild_changes_handles_multiple_punctuation_fixes(local_module):
    """Несколько добавленных запятых в разных местах → несколько пунктов."""
    raw = "А именно отдел подготовил отчёт но никто не стал его читать"
    corrected = "А именно, отдел подготовил отчёт, но никто не стал его читать"
    entries = local_module._rebuild_changes_from_diff(raw, corrected)
    # Минимум 2 пункта (две запятые в разных местах)
    assert len(entries) >= 2
    # Все «было» — substring исходника
    for entry in entries:
        before = entry.split("»")[0].lstrip("«")
        assert before in raw, f"Пункт {entry!r}: «{before}» нет в raw_text"


def test_local_has_real_change_items(local_module):
    """Заглушка «Ошибок не найдено» не считается содержательным пунктом."""
    stub = (
        "===CORRECTED===\n"
        "текст\n"
        "===CHANGES===\n"
        "1. Ошибок не найдено. Текст соответствует нормам.\n"
        "===END==="
    )
    assert local_module._has_real_change_items(stub) is False

    real = (
        "===CORRECTED===\n"
        "текст\n"
        "===CHANGES===\n"
        "1. «X» → «Y» | замена\n"
        "===END==="
    )
    assert local_module._has_real_change_items(real) is True


def test_local_extract_corrected_body(local_module):
    """Извлекаем тело CORRECTED без посторонних маркеров."""
    text = (
        "===CORRECTED===\n"
        "Главным управлением проверяется информация.\n"
        "===CHANGES===\n"
        "1. дроп\n"
        "===END==="
    )
    body = local_module._extract_corrected_body(text)
    assert body == "Главным управлением проверяется информация."


def test_local_suggest_reconstructs_changes_when_model_lies(local_module, monkeypatch):
    """Интеграция: модель отдаёт правильный CORRECTED (с новыми запятыми),
    но в CHANGES выдумывает «безопасностей» (которого в тексте нет). После
    `_drop_changes_not_in_text` пункт выкидывается. Сервер должен реконструировать
    CHANGES из diff(raw_text, CORRECTED) и отдать клиенту валидный список."""
    from fastapi.testclient import TestClient

    raw_input = (
        "Главным управлением собственной безопасности Федеральной службы "
        "в ходе выполнения задач по обеспечению собственной безопасности "
        "проверяется информация о противоправных действиях."
    )

    async def fake_call_ollama(messages):
        # Модель добавила запятые (правильно) НО в CHANGES выдумала пункт
        return (
            "===CORRECTED===\n"
            "Главным управлением собственной безопасности Федеральной службы, "
            "в ходе выполнения задач, по обеспечению собственной безопасности "
            "проверяется информация о противоправных действиях.\n"
            "===CHANGES===\n"
            "1. «безопасностей» → «безопасности» | ошибка в окончании слова\n"
            "===END==="
        )

    monkeypatch.setattr(local_module, "call_ollama", fake_call_ollama)
    client = TestClient(local_module.app)
    files = {
        "text": ("t.txt", io.BytesIO(raw_input.encode("utf-8")), "text/plain"),
        "context": ("c.txt", io.BytesIO(b""), "text/plain"),
    }
    r = client.post("/suggest", files=files)
    assert r.status_code == 200
    body = r.text
    # Галлюцинированный пункт «безопасностей» дропнут
    assert "«безопасностей»" not in body
    # «Ошибок не найдено» НЕ должно появиться: сервер реконструировал из diff
    assert "Ошибок не найдено" not in body
    # Реконструированные пункты появились (хотя бы один с автоправкой)
    assert "автоправка по diff" in body
    # Клиентский InStr найдёт хотя бы один из реконструированных «было»
    # в исходнике (это инвариант — берём substring raw_input)
    pairs = re.findall(r"«([^»]+)»\s*→", body)
    assert pairs, "Должны быть реконструированные пункты"
    found_at_least_one = any(p in raw_input for p in pairs)
    assert found_at_least_one, f"Ни одно «было» не найдено в raw_input: {pairs}"


def test_local_replace_changes_block_with_rebuilt_entries(local_module):
    """После реконструкции CHANGES целиком заменён, рамки целы."""
    text = (
        "===CORRECTED===\n"
        "новый текст\n"
        "===CHANGES===\n"
        "1. Ошибок не найдено. Текст соответствует нормам.\n"
        "===END==="
    )
    entries = ["«старое слово» → «новое слово» | автоправка"]
    out = local_module._replace_changes_block(text, entries)
    assert "===CORRECTED===" in out
    assert "новый текст" in out
    assert "1. «старое слово» → «новое слово»" in out
    assert "Ошибок не найдено" not in out
    assert out.endswith("===END===")


def test_local_strip_thinking_preserves_non_thinking(local_module):
    """Если в ответе нет ни <think>, ни ===CORRECTED=== — возвращаем как есть."""
    raw = "ПроизвольныйТекстБезМаркеров"
    out = local_module._strip_thinking(raw)
    assert out == "ПроизвольныйТекстБезМаркеров"


def test_local_empty_text_returns_error(local_module):
    from fastapi.testclient import TestClient
    client = TestClient(local_module.app)
    files = {
        "text": ("t.txt", io.BytesIO(b"   "), "text/plain"),
        "context": ("c.txt", io.BytesIO(b""), "text/plain"),
    }
    r = client.post("/suggest", files=files)
    assert r.status_code == 200
    assert r.text.startswith("ОШИБКА")


def test_cloud_suggest_with_mocked_openrouter(cloud_module, monkeypatch):
    from fastapi.testclient import TestClient

    async def fake_call_model(messages, model):
        return (
            "===CORRECTED===\nок\n"
            "===CHANGES===\n"
            "1. Ошибок не найдено.\n"
            "===END==="
        )

    monkeypatch.setattr(cloud_module, "call_model", fake_call_model)
    client = TestClient(cloud_module.app)
    files = {
        "text": ("t.txt", io.BytesIO("пример текста".encode("utf-8")), "text/plain"),
        "context": ("c.txt", io.BytesIO(b""), "text/plain"),
    }
    r = client.post("/suggest", files=files)
    assert r.status_code == 200
    assert "===CORRECTED===" in r.text


def test_cloud_metrics(cloud_module):
    from fastapi.testclient import TestClient
    client = TestClient(cloud_module.app)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.json()["server"] == "cloud"


def test_cloud_missing_key(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("AUDIT_DB", str(tmp_path / "audit.sqlite"))

    for m in list(sys.modules):
        if m.startswith(("shared", "main")):
            sys.modules.pop(m, None)

    cloud_dir = ROOT / "server" / "cloud"
    sys.path.insert(0, str(cloud_dir))
    try:
        module = importlib.import_module("main")
        from fastapi.testclient import TestClient
        client = TestClient(module.app)
        files = {
            "text": ("t.txt", io.BytesIO("x".encode("utf-8")), "text/plain"),
            "context": ("c.txt", io.BytesIO(b""), "text/plain"),
        }
        r = client.post("/suggest", files=files)
        assert "ОШИБКА" in r.text and "OPENROUTER_API_KEY" in r.text
    finally:
        sys.path.remove(str(cloud_dir))


# ─── v1.8a / v1.8b tests ───────────────────────────────────────────────


def test_v18_dict_list_endpoint(local_module, monkeypatch, tmp_path):
    """GET /dict/list возвращает пустой список из свежего словаря."""
    from fastapi.testclient import TestClient
    # Изолируем словарь через AI_SUGGESTER_USER_DICT_PATH
    dict_path = tmp_path / "test_dict.json"
    monkeypatch.setattr(local_module, "_user_dict",
                        local_module.shared.user_dict.UserDictionary(dict_path)
                        if hasattr(local_module, "shared") else None)
    # Импорт через модуль если выше не сработал
    from shared.user_dict import UserDictionary
    local_module._user_dict = UserDictionary(dict_path)
    client = TestClient(local_module.app)
    r = client.get("/dict/list")
    assert r.status_code == 200
    assert r.json() == {"words": []}


def test_v18_dict_add_endpoint(local_module, monkeypatch, tmp_path):
    """POST /dict/add добавляет слово, GET /dict/list его возвращает."""
    from fastapi.testclient import TestClient
    from shared.user_dict import UserDictionary
    dict_path = tmp_path / "test_dict.json"
    local_module._user_dict = UserDictionary(dict_path)
    client = TestClient(local_module.app)
    r = client.post("/dict/add", json={"word": "ЦСН"})
    assert r.status_code == 200
    assert r.json() == {"added": True, "total": 1}
    r2 = client.post("/dict/add", json={"word": "ЦСН"})
    assert r2.json() == {"added": False, "total": 1}
    r3 = client.get("/dict/list")
    assert r3.json() == {"words": ["ЦСН"]}


def test_v18_dict_add_validates_input(local_module, monkeypatch, tmp_path):
    """POST /dict/add отвергает мусор."""
    from fastapi.testclient import TestClient
    from shared.user_dict import UserDictionary
    dict_path = tmp_path / "test_dict.json"
    local_module._user_dict = UserDictionary(dict_path)
    client = TestClient(local_module.app)
    # Пустое слово
    r = client.post("/dict/add", json={"word": ""})
    assert r.status_code == 400
    # Не строка
    r2 = client.post("/dict/add", json={"word": 123})
    assert r2.status_code == 400
    # С запрещёнными символами
    r3 = client.post("/dict/add", json={"word": "<script>"})
    assert r3.status_code == 400


def test_v18_dict_remove_endpoint(local_module, monkeypatch, tmp_path):
    """POST /dict/remove удаляет слово."""
    from fastapi.testclient import TestClient
    from shared.user_dict import UserDictionary
    dict_path = tmp_path / "test_dict.json"
    local_module._user_dict = UserDictionary(dict_path)
    client = TestClient(local_module.app)
    client.post("/dict/add", json={"word": "ЦСН"})
    r = client.post("/dict/remove", json={"word": "ЦСН"})
    assert r.status_code == 200
    assert r.json() == {"removed": True, "total": 0}
    r2 = client.post("/dict/remove", json={"word": "ЦСН"})
    assert r2.json() == {"removed": False, "total": 0}


def test_v18_dict_disabled_returns_503(local_module, monkeypatch):
    """Если USER_DICT_ENABLED=false (или _user_dict=None) — 503."""
    from fastapi.testclient import TestClient
    monkeypatch.setattr(local_module, "_user_dict", None)
    client = TestClient(local_module.app)
    r = client.get("/dict/list")
    assert r.status_code == 503
    r2 = client.post("/dict/add", json={"word": "ЦСН"})
    assert r2.status_code == 503


def test_v18_drop_user_dict_changes_no_op_when_empty(local_module):
    """_drop_user_dict_changes — no-op если словарь пустой."""
    from shared.user_dict import UserDictionary
    import tempfile
    import os as _os
    fd, path = tempfile.mkstemp(suffix=".json")
    _os.close(fd)
    _os.unlink(path)
    try:
        local_module._user_dict = UserDictionary(__import__("pathlib").Path(path))
        text = (
            "===CORRECTED===\nтекст\n===CHANGES===\n"
            "1. «ЦСН» → «ЦНС» | опечатка\n===END==="
        )
        out = local_module._drop_user_dict_changes(text)
        assert out == text  # no-op
    finally:
        if _os.path.exists(path):
            _os.unlink(path)


def test_v18_drop_user_dict_changes_drops_whitelisted(local_module, tmp_path):
    """_drop_user_dict_changes дропает пункты, в которых модель «исправляет»
    whitelisted-термин."""
    from shared.user_dict import UserDictionary
    dict_path = tmp_path / "ud.json"
    local_module._user_dict = UserDictionary(dict_path)
    local_module._user_dict.add("ЦСН")
    text = (
        "===CORRECTED===\nтекст\n===CHANGES===\n"
        "1. «ЦСН» → «ЦНС» | опечатка\n"
        "2. «работ» → «работы» | согласование\n"
        "===END==="
    )
    out = local_module._drop_user_dict_changes(text)
    assert "«ЦСН» → «ЦНС»" not in out
    assert "«работ» → «работы»" in out


def test_v18_drop_user_dict_changes_case_insensitive(local_module, tmp_path):
    """Whitelist должен быть case-insensitive."""
    from shared.user_dict import UserDictionary
    dict_path = tmp_path / "ud.json"
    local_module._user_dict = UserDictionary(dict_path)
    local_module._user_dict.add("ЦСН")
    text = (
        "===CORRECTED===\nтекст\n===CHANGES===\n"
        "1. «цсн» → «ЦНС» | опечатка\n===END==="
    )
    out = local_module._drop_user_dict_changes(text)
    # «цсн» (lowercase) — то же что «ЦСН» в whitelist (case-insensitive)
    assert "опечатка" not in out


def test_v18_enrich_changes_with_detector_adds_kvartalakh(local_module):
    """enrich-функция добавляет «во 2-м кварталах» если детектор нашёл."""
    text = (
        "===CORRECTED===\n"
        "Во 2-м кварталах планируется направление материалов проверки.\n"
        "===CHANGES===\n"
        "1. Ошибок не найдено. Текст соответствует нормам.\n"
        "===END==="
    )
    raw = "Во 2-м кварталах планируется направление материалов проверки."
    out = local_module._enrich_changes_with_detector(text, raw)
    # Должна появиться правка «кварталах» → «квартале»
    assert "кварталах" in out
    assert "квартале" in out
    # Должно быть в CHANGES блоке
    changes_block = out.split("===CHANGES===")[1].split("===END===")[0]
    assert "«кварталах» → «квартале»" in changes_block


def test_v18_enrich_changes_dedupes_existing(local_module):
    """Если модель уже отдала пункт с тем же before — детектор не дублирует."""
    text = (
        "===CORRECTED===\n"
        "Во 2-м квартале планируется.\n"
        "===CHANGES===\n"
        "1. «кварталах» → «квартале» | согласование с числительным\n"
        "===END==="
    )
    raw = "Во 2-м кварталах планируется."
    out = local_module._enrich_changes_with_detector(text, raw)
    changes_block = out.split("===CHANGES===")[1].split("===END===")[0]
    # Пункт встречается ровно один раз
    assert changes_block.count("«кварталах»") == 1


def test_v18_enrich_changes_no_op_if_disabled(local_module, monkeypatch):
    """Если детектор отключён — функция no-op."""
    monkeypatch.setattr(local_module, "_morph_detector", None)
    text = (
        "===CORRECTED===\nтекст\n===CHANGES===\n1. Ошибок не найдено.\n===END==="
    )
    raw = "Во 2-м кварталах ошибок."
    out = local_module._enrich_changes_with_detector(text, raw)
    assert out == text


def test_v18_metrics_includes_detector_and_dict_state(local_module, tmp_path):
    """GET /metrics возвращает morph_detector_enabled и user_dict_size."""
    from fastapi.testclient import TestClient
    from shared.user_dict import UserDictionary
    dict_path = tmp_path / "ud.json"
    local_module._user_dict = UserDictionary(dict_path)
    local_module._user_dict.add("ЦСН")
    client = TestClient(local_module.app)
    r = client.get("/metrics")
    data = r.json()
    assert "morph_detector_enabled" in data
    assert "user_dict_enabled" in data
    assert data["user_dict_size"] == 1


def test_v18_full_pipeline_meropriyatie_and_kvartalakh(local_module, monkeypatch):
    """E2E: модель ничего не правит, детектор находит «мероприятия» и «кварталах»."""
    from fastapi.testclient import TestClient

    async def fake_call_ollama(messages):
        # Модель не нашла ошибок (как в проде — пропустила «кварталах»)
        return (
            "===CORRECTED===\n"
            "Проверочное мероприятия. Во 2-м кварталах планируется.\n"
            "===CHANGES===\n"
            "1. Ошибок не найдено. Текст соответствует нормам.\n"
            "===END==="
        )

    monkeypatch.setattr(local_module, "call_ollama", fake_call_ollama)
    client = TestClient(local_module.app)
    raw = "Проверочное мероприятия. Во 2-м кварталах планируется."
    files = {
        "text": ("t.txt", io.BytesIO(raw.encode("utf-8")), "text/plain"),
        "context": ("c.txt", io.BytesIO(b""), "text/plain"),
    }
    r = client.post("/suggest", files=files)
    assert r.status_code == 200
    body = r.text
    # Детектор добавил минимум одну правку
    assert "кварталах" in body or "мероприятия" in body
    changes = body.split("===CHANGES===")[1].split("===END===")[0]
    has_meropr = "«мероприятия»" in changes
    has_kvar = "«кварталах»" in changes
    assert has_meropr or has_kvar, f"Детектор не сработал. CHANGES:\n{changes}"


# ═══════════════════════════════════════════════════════════════════════════
# v1.8c: тесты sage-95m post-валидатора (filter_changes_with_sage)
# ═══════════════════════════════════════════════════════════════════════════
# Реальная sage-fredt5 модель НЕ загружается — тесты подменяют валидатор
# мок-объектом с фиксированным judge() поведением.

class _MockSageValidator:
    """Мок-валидатор: возвращает заранее заданные verdict'ы по before-строке.

    is_available=True, чтобы пайплайн прошёл через _filter_changes_with_sage.
    sage_text возвращается фиксированный — пайплайн им не пользуется напрямую,
    кроме как условием sage_text != raw_text (чтобы не вернуть text сразу).
    """
    def __init__(self, *, verdicts_by_before, sage_text, domain="admin",
                 mode="enforce", categories=()):
        from server.shared.sage_validator import SageConfig
        self.config = SageConfig(
            enabled=True, mode=mode, domain=domain, categories=categories,
            model_name="mock", device="cpu", max_input_len=512, warmup=False,
        )
        self._verdicts = verdicts_by_before
        self._sage_text = sage_text

    def is_available(self):
        return True

    def correct(self, text):
        return self._sage_text

    def judge(self, before, after, sage_text):
        from server.shared.sage_validator import VERDICT_UNKNOWN
        return self._verdicts.get(before, VERDICT_UNKNOWN)

    def category_matches(self, category):
        if not self.config.categories:
            return True
        cat_lower = (category or "").lower()
        return any(c in cat_lower for c in self.config.categories)

    def should_drop(self, verdict, *, category=""):
        from server.shared.sage_validator import (
            VERDICT_AGREE, VERDICT_DISAGREE, VERDICT_UNKNOWN,
        )
        if self.config.mode == "dryrun":
            return False
        if not self.category_matches(category):
            return False
        if verdict == VERDICT_DISAGREE:
            return True
        if verdict == VERDICT_UNKNOWN and self.config.domain == "general":
            return True
        return False


def test_v18c_sage_drops_disagree_in_admin(local_module, monkeypatch):
    """admin-режим: пункт DISAGREE должен быть отфильтрован."""
    from server.shared.sage_validator import VERDICT_DISAGREE, VERDICT_AGREE
    mock = _MockSageValidator(
        verdicts_by_before={"мероприятие": VERDICT_DISAGREE, "кварталах": VERDICT_AGREE},
        sage_text="Во 2-м квартале проведено мероприятие.",
        domain="admin",
    )
    monkeypatch.setattr(local_module, "_sage_validator", mock)
    raw_in = (
        "===CORRECTED===\n"
        "Во 2-м квартале проведено мероприятия.\n"
        "===CHANGES===\n"
        "1. «кварталах» → «квартале» | согласование\n"
        "2. «мероприятие» → «мероприятия» | согласование\n"
        "===END===\n"
    )
    out = local_module._filter_changes_with_sage(
        raw_in, "Во 2-м кварталах проведено мероприятие."
    )
    # Sage не согласен с правкой «мероприятие→мероприятия» — она должна быть дропнута
    assert "«мероприятие»" not in out or "мероприятие→мероприятия" not in out
    # Sage согласен с «кварталах→квартале» — правка должна остаться
    assert "«кварталах»" in out and "«квартале»" in out
    # CORRECTED откатил «мероприятия» обратно в «мероприятие»
    corrected_body = out.split("===CORRECTED===")[1].split("===CHANGES===")[0]
    assert "мероприятие" in corrected_body
    assert "мероприятия" not in corrected_body


def test_v18c_sage_keeps_unknown_in_admin(local_module, monkeypatch):
    """admin-режим: UNKNOWN не дропается, recall важнее."""
    from server.shared.sage_validator import VERDICT_UNKNOWN, VERDICT_AGREE
    mock = _MockSageValidator(
        verdicts_by_before={"редкоеслово": VERDICT_UNKNOWN, "кварталах": VERDICT_AGREE},
        sage_text="Совсем другой текст от sage'a.",
        domain="admin",
    )
    monkeypatch.setattr(local_module, "_sage_validator", mock)
    raw_in = (
        "===CORRECTED===\n"
        "Во 2-м квартале редкоеновое слово.\n"
        "===CHANGES===\n"
        "1. «кварталах» → «квартале» | согласование\n"
        "2. «редкоеслово» → «редкоеновое» | редкая правка\n"
        "===END===\n"
    )
    out = local_module._filter_changes_with_sage(
        raw_in, "Во 2-м кварталах редкоеслово слово."
    )
    # Обе правки остались, потому что admin не дропает UNKNOWN
    assert "«редкоеслово»" in out
    assert "«кварталах»" in out


def test_v18c_sage_drops_unknown_in_general(local_module, monkeypatch):
    """general-режим: UNKNOWN тоже дропается."""
    from server.shared.sage_validator import VERDICT_UNKNOWN, VERDICT_AGREE
    mock = _MockSageValidator(
        verdicts_by_before={"редкоеслово": VERDICT_UNKNOWN, "кварталах": VERDICT_AGREE},
        sage_text="Совсем другой текст от sage'a.",
        domain="general",
    )
    monkeypatch.setattr(local_module, "_sage_validator", mock)
    raw_in = (
        "===CORRECTED===\n"
        "Во 2-м квартале редкоеновое слово.\n"
        "===CHANGES===\n"
        "1. «кварталах» → «квартале» | согласование\n"
        "2. «редкоеслово» → «редкоеновое» | редкая правка\n"
        "===END===\n"
    )
    out = local_module._filter_changes_with_sage(
        raw_in, "Во 2-м кварталах редкоеслово слово."
    )
    # «редкоеслово» дропнут (UNKNOWN в general), «кварталах» оставлен (AGREE)
    assert "«редкоеслово»" not in out
    assert "«кварталах»" in out


def test_v18c_sage_noop_when_disabled(local_module, monkeypatch):
    """SAGE_VALIDATOR_ENABLED=false → пайплайн возвращает text как есть."""
    # _sage_validator = None означает «недоступен/отключён»
    monkeypatch.setattr(local_module, "_sage_validator", None)
    raw_in = (
        "===CORRECTED===\n"
        "Что-то.\n"
        "===CHANGES===\n"
        "1. «a» → «б» | проверка\n"
        "===END===\n"
    )
    out = local_module._filter_changes_with_sage(raw_in, "Что-то.")
    assert out == raw_in


def test_v18c_sage_noop_when_sage_text_unchanged(local_module, monkeypatch):
    """Если sage не нашёл ошибок (вернул тот же raw_text), не дропаем НИЧЕГО.

    Это критично: sage-95m может пропускать ошибки, нельзя по его «всё ОК»
    обнулять CHANGES от T-lite + детектора.
    """
    raw = "Текст без изменений."
    mock = _MockSageValidator(
        verdicts_by_before={"a": "disagree"},
        sage_text=raw,  # sage вернул ТОТ ЖЕ текст
        domain="admin",
    )
    monkeypatch.setattr(local_module, "_sage_validator", mock)
    raw_in = (
        "===CORRECTED===\n"
        "Текст с правкой.\n"
        "===CHANGES===\n"
        "1. «a» → «б» | проверка\n"
        "===END===\n"
    )
    out = local_module._filter_changes_with_sage(raw_in, raw)
    # Ничего не дропнуто — text возвращён без изменений
    assert "«a»" in out


def test_v18c_sage_handles_no_changes_block(local_module, monkeypatch):
    """Если CHANGES блока нет — no-op, без падения."""
    mock = _MockSageValidator(
        verdicts_by_before={}, sage_text="другой", domain="admin",
    )
    monkeypatch.setattr(local_module, "_sage_validator", mock)
    # raw без маркера CHANGES
    out = local_module._filter_changes_with_sage("Просто текст.", "raw")
    assert out == "Просто текст."


def test_v18c_sage_dryrun_never_drops(local_module, monkeypatch):
    """dryrun-режим: даже DISAGREE-правки остаются в выводе (только логи).

    Это default-режим. Прод включает sage сначала в dryrun, чтобы собрать
    verdict'ы, и только после анализа логов переключает в enforce.
    """
    from server.shared.sage_validator import VERDICT_DISAGREE
    mock = _MockSageValidator(
        verdicts_by_before={"мероприятие": VERDICT_DISAGREE},
        sage_text="Другой текст — sage что-то править.",
        domain="admin",
        mode="dryrun",  # ← ключевое отличие
    )
    monkeypatch.setattr(local_module, "_sage_validator", mock)
    raw_in = (
        "===CORRECTED===\n"
        "Проведено мероприятия.\n"
        "===CHANGES===\n"
        "1. «мероприятие» → «мероприятия» | согласование\n"
        "===END===\n"
    )
    out = local_module._filter_changes_with_sage(raw_in, "Проведено мероприятие.")
    # В dryrun ничего не дропаем
    assert "«мероприятие»" in out


def test_v18c_sage_category_filter_keeps_non_orthography(local_module, monkeypatch):
    """enforce-режим с categories=("орфограф",): согласование НЕ дропается,
    даже если verdict=DISAGREE — sage обучена на орфографии, для других
    классов её мнение ненадёжно."""
    from server.shared.sage_validator import VERDICT_DISAGREE
    mock = _MockSageValidator(
        verdicts_by_before={"кварталах": VERDICT_DISAGREE},
        sage_text="Что-то другое.",
        domain="admin",
        mode="enforce",
        categories=("орфограф",),
    )
    monkeypatch.setattr(local_module, "_sage_validator", mock)
    raw_in = (
        "===CORRECTED===\n"
        "Во 2-м квартале.\n"
        "===CHANGES===\n"
        "1. «кварталах» → «квартале» | согласование\n"
        "===END===\n"
    )
    out = local_module._filter_changes_with_sage(raw_in, "Во 2-м кварталах.")
    # Категория «согласование» не входит в фильтр («орфограф»), правка остаётся
    assert "«кварталах»" in out


def test_v18c_sage_category_filter_drops_orthography(local_module, monkeypatch):
    """enforce-режим с categories=("орфограф",): орфо-правки дропаются."""
    from server.shared.sage_validator import VERDICT_DISAGREE
    mock = _MockSageValidator(
        verdicts_by_before={"опечтка": VERDICT_DISAGREE},
        sage_text="Без слова опечтка.",
        domain="admin",
        mode="enforce",
        categories=("орфограф",),
    )
    monkeypatch.setattr(local_module, "_sage_validator", mock)
    raw_in = (
        "===CORRECTED===\n"
        "Слово опечатка.\n"
        "===CHANGES===\n"
        "1. «опечтка» → «опечатка» | орфография — пропущена буква\n"
        "===END===\n"
    )
    out = local_module._filter_changes_with_sage(raw_in, "Слово опечтка.")
    # Категория содержит «орфограф», sage DISAGREE → правка дропнута
    assert "«опечтка»" not in out


# ============================================================
# v1.8.4: _complete_changes_from_corrected — CHANGES↔CORRECTED desync
# ============================================================


def test_v184_completes_missing_agreement_change(local_module):
    """Реальный прод-кейс v1.8c прогона (05.05.2026): T-lite склеила две
    правки в один пункт. CHANGES перечисляет только «ремонтова → ремонта»
    (орфография), но CORRECTED содержит ещё и «капитальных → капитального»
    (согласование). Функция должна добавить недостающую правку."""
    raw = "выполнением капитальных ремонтова помещений"
    text = (
        "===CORRECTED===\n"
        "выполнением капитального ремонта помещений\n"
        "===CHANGES===\n"
        "1. «ремонтова» → «ремонта» | орфография — пропущена буква «н»\n"
        "===END===\n"
    )
    out = local_module._complete_changes_from_corrected(text, raw)
    # Старый пункт остался
    assert "«ремонтова» → «ремонта»" in out
    # И добавился новый — какая-то правка вокруг «капитальных»
    # (точная формулировка зависит от _expand_word_context).
    assert "капитальных" in out
    assert "капитального" in out


def test_v184_noop_when_changes_fully_cover_corrected(local_module):
    """Если применение CHANGES к raw даёт ровно CORRECTED — функция
    не должна ничего менять."""
    raw = "ремонтова помещений"
    text = (
        "===CORRECTED===\n"
        "ремонта помещений\n"
        "===CHANGES===\n"
        "1. «ремонтова» → «ремонта» | орфография\n"
        "===END===\n"
    )
    out = local_module._complete_changes_from_corrected(text, raw)
    assert out == text


def test_v184_noop_when_raw_equals_corrected(local_module):
    """Если CORRECTED == raw_text (ошибок нет), функция — no-op."""
    raw = "Текст без ошибок."
    text = (
        "===CORRECTED===\n"
        "Текст без ошибок.\n"
        "===CHANGES===\n"
        "1. Ошибок не найдено. Текст соответствует нормам.\n"
        "===END===\n"
    )
    out = local_module._complete_changes_from_corrected(text, raw)
    assert out == text


def test_v184_noop_when_changes_empty(local_module):
    """Пустой CHANGES + CORRECTED == raw → no-op."""
    raw = "Текст."
    text = (
        "===CORRECTED===\n"
        "Текст.\n"
        "===CHANGES===\n"
        "===END===\n"
    )
    out = local_module._complete_changes_from_corrected(text, raw)
    assert out == text


def test_v184_noop_when_no_corrected_block(local_module):
    """Если в text нет ===CORRECTED=== — no-op."""
    raw = "raw text"
    text = "просто текст без маркеров"
    out = local_module._complete_changes_from_corrected(text, raw)
    assert out == text


def test_v184_noop_when_no_changes_block(local_module):
    """Если в text нет ===CHANGES=== — no-op."""
    raw = "raw text"
    text = "===CORRECTED===\ncorrected\n"
    out = local_module._complete_changes_from_corrected(text, raw)
    assert out == text


def test_v184_dedupes_already_existing_before(local_module):
    """Если diff даёт правку с тем же `before`, что уже в CHANGES —
    не дублируем."""
    raw = "опечтка слово"
    text = (
        "===CORRECTED===\n"
        "опечатка слово\n"
        "===CHANGES===\n"
        "1. «опечтка» → «опечатка» | орфография\n"
        "===END===\n"
    )
    out = local_module._complete_changes_from_corrected(text, raw)
    # Симуляция применения CHANGES даёт ровно CORRECTED, новых правок нет
    assert out == text


def test_v184_strips_stub_when_adding_real_changes(local_module):
    """Если CHANGES содержит только стаб «Ошибок не найдено», но
    CORRECTED отличается от raw — стаб должен быть затерт, новые
    правки добавлены."""
    raw = "Это ошибк."
    text = (
        "===CORRECTED===\n"
        "Это ошибка.\n"
        "===CHANGES===\n"
        "1. Ошибок не найдено. Текст соответствует нормам.\n"
        "===END===\n"
    )
    out = local_module._complete_changes_from_corrected(text, raw)
    # Стаб исчез
    assert "Ошибок не найдено" not in out
    # И появилась реальная правка
    assert "ошибк" in out
    assert "ошибка" in out


def test_v184_skips_missing_before_in_simulation(local_module):
    """Если `before` пункта CHANGES не найдён в raw_text — пропускаем
    его при симуляции (это уже отфильтровано _drop_changes_not_in_text,
    но защищаемся от багов). Diff всё равно построится правильно."""
    raw = "Реальный текст."
    text = (
        "===CORRECTED===\n"
        "Реальный текст.\n"
        "===CHANGES===\n"
        "1. «несуществующий» → «фрагмент» | мусор\n"
        "===END===\n"
    )
    out = local_module._complete_changes_from_corrected(text, raw)
    # simulated == raw == CORRECTED.strip() → no-op
    assert out == text


def test_v184_preserves_other_block_structure(local_module):
    """Хвост после ===END=== должен быть сохранён без изменений."""
    raw = "ремонтова"
    text = (
        "===CORRECTED===\n"
        "ремонта\n"
        "===CHANGES===\n"
        "1. «ремонтова» → «ремонта» | орфография\n"
        "===END===\n"
        "Дополнительный хвост\n"
    )
    out = local_module._complete_changes_from_corrected(text, raw)
    assert "Дополнительный хвост" in out


# ─── v2.0-a: LLM_PRESET A/B/C тесты ───────────────────────────────────


def _load_local_server_with_preset(monkeypatch, tmp_path, preset, model_override=None):
    """Загружает local-сервер с заданным LLM_PRESET. Если model_override
    задан — также устанавливает MODEL_NAME (для проверки приоритета).
    """
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("AUDIT_DB", str(tmp_path / "audit.sqlite"))
    monkeypatch.setenv("RAG_ENABLED", "false")
    monkeypatch.setenv("LLM_PRESET", preset)
    if model_override is not None:
        monkeypatch.setenv("MODEL_NAME", model_override)
    else:
        monkeypatch.delenv("MODEL_NAME", raising=False)
    for m in list(sys.modules):
        if m.startswith(("shared", "main")):
            sys.modules.pop(m, None)
    local_dir = ROOT / "server" / "local"
    sys.path.insert(0, str(local_dir))
    module = importlib.import_module("main")
    yield module
    sys.path.remove(str(local_dir))
    for m in list(sys.modules):
        if m.startswith(("shared", "main")):
            sys.modules.pop(m, None)


@pytest.fixture
def local_module_preset_a(monkeypatch, tmp_path):
    yield from _load_local_server_with_preset(monkeypatch, tmp_path, "A")


@pytest.fixture
def local_module_preset_b(monkeypatch, tmp_path):
    yield from _load_local_server_with_preset(monkeypatch, tmp_path, "B")


@pytest.fixture
def local_module_preset_c(monkeypatch, tmp_path):
    yield from _load_local_server_with_preset(monkeypatch, tmp_path, "C")


@pytest.fixture
def local_module_preset_unknown(monkeypatch, tmp_path):
    yield from _load_local_server_with_preset(monkeypatch, tmp_path, "Z")


@pytest.fixture
def local_module_preset_with_override(monkeypatch, tmp_path):
    yield from _load_local_server_with_preset(
        monkeypatch, tmp_path, "A", model_override="qwen3:30b-a3b"
    )


def test_v20a_preset_a_default_tlite(local_module_preset_a):
    """Preset A → MODEL_NAME = t-tech/T-lite (baseline default)."""
    assert local_module_preset_a.LLM_PRESET == "A"
    assert local_module_preset_a.MODEL_NAME == "t-tech/T-lite-it-2.1:q4_K_M"


def test_v20a_preset_b_yandex(local_module_preset_b):
    """Preset B → MODEL_NAME = YandexGPT-5-Lite-8B GGUF."""
    assert local_module_preset_b.LLM_PRESET == "B"
    assert "yandex" in local_module_preset_b.MODEL_NAME.lower()
    assert "YandexGPT-5-Lite" in local_module_preset_b.MODEL_NAME


def test_v20a_preset_c_gigachat(local_module_preset_c):
    """Preset C → MODEL_NAME = GigaChat-3.1-Lightning."""
    assert local_module_preset_c.LLM_PRESET == "C"
    assert "GigaChat-3.1-Lightning" in local_module_preset_c.MODEL_NAME
    assert "ai-sage" in local_module_preset_c.MODEL_NAME.lower()


def test_v20a_preset_unknown_falls_back_to_a(local_module_preset_unknown):
    """Незнакомый preset → fallback на preset A (T-lite). Сервер
    остаётся работоспособным, не падает с KeyError на старте.
    """
    assert local_module_preset_unknown.LLM_PRESET == "Z"
    assert local_module_preset_unknown.MODEL_NAME == "t-tech/T-lite-it-2.1:q4_K_M"


def test_v20a_explicit_model_name_overrides_preset(
    local_module_preset_with_override,
):
    """Явный MODEL_NAME имеет приоритет над LLM_PRESET. Позволяет
    тестировать произвольные модели без правки кода presets.
    """
    mod = local_module_preset_with_override
    assert mod.LLM_PRESET == "A"
    # При том MODEL_NAME должен быть override-значением, не T-lite
    assert mod.MODEL_NAME == "qwen3:30b-a3b"


def test_v20a_metrics_includes_preset(local_module_preset_b):
    """/metrics возвращает llm_preset и llm_preset_description."""
    from fastapi.testclient import TestClient
    client = TestClient(local_module_preset_b.app)
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert data["llm_preset"] == "B"
    assert "yandex" in data["llm_preset_description"].lower() or "Yandex" in data["llm_preset_description"]
    assert data["model"] == local_module_preset_b.MODEL_NAME


def test_v20a_presets_dict_structure(local_module_preset_a):
    """Все presets имеют MODEL_NAME и DESCRIPTION (защита от опечаток
    при добавлении новых presets)."""
    presets = local_module_preset_a.LLM_PRESETS
    assert set(presets.keys()) >= {"A", "B", "C"}
    for key, cfg in presets.items():
        assert "MODEL_NAME" in cfg, f"preset {key} missing MODEL_NAME"
        assert "DESCRIPTION" in cfg, f"preset {key} missing DESCRIPTION"
        assert cfg["MODEL_NAME"], f"preset {key} MODEL_NAME пустой"
        assert cfg["DESCRIPTION"], f"preset {key} DESCRIPTION пустое"


# ─── v2.0-b: LanguageTool интеграция ──────────────────────────────────


def _load_local_server_with_lt(monkeypatch, tmp_path, lt_enabled=True):
    """Загружает local-сервер с активным LANGUAGETOOL_ENABLED.

    Используем `transport`-параметр LanguageToolClient (а не глобальный
    monkeypatch httpx.Client — это бы ломало starlette.TestClient,
    который наследуется от httpx.Client). Перехватываем функцию
    `get_languagetool_client` чтобы инжектировать MockTransport.
    """
    import httpx

    def handler(request):
        if request.url.path == "/v2/languages":
            return httpx.Response(200, json=[
                {"longCode": "ru-RU", "code": "ru", "name": "Russian"},
            ])
        if request.url.path == "/v2/check":
            return httpx.Response(200, json={
                "matches": [
                    {
                        "message": "Используйте длинное тире",
                        "offset": 13, "length": 1,
                        "replacements": [{"value": "—"}],
                        "rule": {
                            "id": "DASH_RULE",
                            "category": {"id": "TYPOGRAPHY", "name": "Типографика"},
                        },
                    },
                ],
            })
        return httpx.Response(404)

    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("AUDIT_DB", str(tmp_path / "audit.sqlite"))
    monkeypatch.setenv("RAG_ENABLED", "false")
    monkeypatch.setenv("LANGUAGETOOL_ENABLED", "true" if lt_enabled else "false")
    monkeypatch.setenv("LANGUAGETOOL_URL", "http://lt-mock:8081")
    monkeypatch.setenv("LANGUAGETOOL_ENABLED_CATEGORIES", "STYLE,TYPOGRAPHY")
    for m in list(sys.modules):
        if m.startswith(("shared", "main")):
            sys.modules.pop(m, None)
    local_dir = ROOT / "server" / "local"
    sys.path.insert(0, str(local_dir))
    # Заворачиваем get_languagetool_client в фабрику с MockTransport
    import shared.languagetool_client as lt_mod
    original_get = lt_mod.get_languagetool_client

    def patched_get(**kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return original_get(**kwargs)

    monkeypatch.setattr(lt_mod, "get_languagetool_client", patched_get)
    lt_mod.reset_client()

    module = importlib.import_module("main")
    yield module
    sys.path.remove(str(local_dir))
    lt_mod.reset_client()
    for m in list(sys.modules):
        if m.startswith(("shared", "main")):
            sys.modules.pop(m, None)


@pytest.fixture
def local_module_lt_enabled(monkeypatch, tmp_path):
    yield from _load_local_server_with_lt(monkeypatch, tmp_path, lt_enabled=True)


@pytest.fixture
def local_module_lt_disabled(monkeypatch, tmp_path):
    yield from _load_local_server_with_lt(monkeypatch, tmp_path, lt_enabled=False)


def test_v20b_lt_disabled_metrics(local_module_lt_disabled):
    """LANGUAGETOOL_ENABLED=false → metrics показывают enabled=false,
    available=false, и url/categories=null."""
    from fastapi.testclient import TestClient
    client = TestClient(local_module_lt_disabled.app)
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert data["languagetool_enabled"] is False
    assert data["languagetool_available"] is False
    assert data["languagetool_url"] is None
    assert data["languagetool_enabled_categories"] is None


def test_v20b_lt_enabled_metrics(local_module_lt_enabled):
    """LANGUAGETOOL_ENABLED=true + LT-mock доступен → metrics показывают
    enabled=true, available=true, url + categories."""
    from fastapi.testclient import TestClient
    client = TestClient(local_module_lt_enabled.app)
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert data["languagetool_enabled"] is True
    assert data["languagetool_available"] is True
    assert data["languagetool_url"] == "http://lt-mock:8081"
    assert data["languagetool_language"] == "ru-RU"
    assert data["languagetool_enabled_categories"] == "STYLE,TYPOGRAPHY"


def test_v20b_lt_enriches_changes(local_module_lt_enabled, monkeypatch):
    """LANGUAGETOOL_ENABLED=true + LT возвращает один match → этот match
    появляется в CHANGES блоке /suggest."""
    from fastapi.testclient import TestClient

    async def fake_call_ollama(messages, *args, **kwargs):
        return (
            "===CORRECTED===\nТест документ - проверка.\n"
            "===CHANGES===\n1. «опечатка» → «опечатки» | орфография\n===END==="
        )

    monkeypatch.setattr(local_module_lt_enabled, "call_ollama", fake_call_ollama)
    client = TestClient(local_module_lt_enabled.app)
    files = {
        "text": ("t.txt", io.BytesIO("Тест документ - проверка.".encode("utf-8")), "text/plain"),
        "context": ("c.txt", io.BytesIO(b""), "text/plain"),
    }
    r = client.post("/suggest", files=files)
    assert r.status_code == 200
    body = r.text
    # T-lite правка (от мока)
    assert "===CHANGES===" in body
    # LT правка ([TYPOGRAPHY] из мока), добавлена в CHANGES
    assert "TYPOGRAPHY" in body or "тире" in body


# ─── v2.1 cloud-mirror tests: parity с local-сервером ─────────────────


def _load_cloud_server_with_env(monkeypatch, tmp_path, **env):
    """Загружает cloud-server с произвольными env-флагами.
    Дефолты: OPENROUTER_API_KEY заполнен, LOG_DIR/AUDIT_DB в tmp.
    Любые дополнительные env-флаги передаются через **env.
    """
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("AUDIT_DB", str(tmp_path / "audit.sqlite"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-key-123456")
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))

    for m in list(sys.modules):
        if m.startswith(("shared", "main")):
            sys.modules.pop(m, None)

    cloud_dir = ROOT / "server" / "cloud"
    sys.path.insert(0, str(cloud_dir))
    module = importlib.import_module("main")
    yield module
    sys.path.remove(str(cloud_dir))
    for m in list(sys.modules):
        if m.startswith(("shared", "main")):
            sys.modules.pop(m, None)


@pytest.fixture
def cloud_module_preset_b(monkeypatch, tmp_path):
    yield from _load_cloud_server_with_env(monkeypatch, tmp_path, CLOUD_PRESET="B")


@pytest.fixture
def cloud_module_preset_unknown(monkeypatch, tmp_path):
    yield from _load_cloud_server_with_env(monkeypatch, tmp_path, CLOUD_PRESET="Z")


@pytest.fixture
def cloud_module_override(monkeypatch, tmp_path):
    yield from _load_cloud_server_with_env(
        monkeypatch, tmp_path,
        OPENROUTER_MODELS="model-x/free,model-y/free",
    )


@pytest.fixture
def cloud_module_dict_disabled(monkeypatch, tmp_path):
    yield from _load_cloud_server_with_env(monkeypatch, tmp_path, USER_DICT_ENABLED="false")


def test_cloud_preset_a_default(cloud_module):
    """Дефолтный preset = A, первая модель — openrouter/free auto-router."""
    assert cloud_module.CLOUD_PRESET == "A"
    assert cloud_module.MODELS[0] == "openrouter/free"
    assert len(cloud_module.MODELS) > 1


def test_cloud_preset_b_qwen(cloud_module_preset_b):
    """CLOUD_PRESET=B → primary = qwen3-next."""
    assert cloud_module_preset_b.CLOUD_PRESET == "B"
    assert cloud_module_preset_b.MODELS[0].startswith("qwen/qwen3-next")


def test_cloud_unknown_preset_falls_back_to_a(cloud_module_preset_unknown):
    """Неизвестный preset → A (мягкая деградация)."""
    assert cloud_module_preset_unknown.CLOUD_PRESET == "A"


def test_cloud_models_override(cloud_module_override):
    """OPENROUTER_MODELS (CSV) перебивает preset."""
    assert cloud_module_override.MODELS == ["model-x/free", "model-y/free"]


def test_cloud_metrics_extended(cloud_module):
    """v2.2 metrics возвращает минимальный набор блоков: rag, dict, preset, audit.

    В v2.2 cloud-сервер сознательно упрощён по сравнению с local: морф-фильтр,
    морф-детектор, sage, LanguageTool и few-shot retrieval удалены, потому что
    сетевая модель сама лучше справляется с этими классами ошибок.
    """
    from fastapi.testclient import TestClient
    client = TestClient(cloud_module.app)
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert data["server"] == "cloud"
    assert "rag_enabled" in data
    assert "user_dict_enabled" in data
    assert "cloud_preset" in data
    assert "cloud_preset_description" in data
    assert "audit" in data
    # В v2.2 этих блоков быть не должно — фильтры удалены
    assert "few_shot_enabled" not in data
    assert "morph_filter_enabled" not in data
    assert "morph_detector_enabled" not in data
    assert "languagetool_enabled" not in data


def test_cloud_dict_list_endpoint(cloud_module):
    """/dict/list возвращает 200 + words, когда USER_DICT_ENABLED=true (дефолт)."""
    from fastapi.testclient import TestClient
    client = TestClient(cloud_module.app)
    r = client.get("/dict/list")
    assert r.status_code == 200
    body = r.json()
    assert "words" in body
    assert isinstance(body["words"], list)


def test_cloud_dict_add_and_remove(cloud_module):
    """/dict/add + /dict/remove работают и /dict/list отражает изменения."""
    from fastapi.testclient import TestClient
    client = TestClient(cloud_module.app)

    # Сначала зачистим словарь (на случай артефактов от других тестов)
    initial = client.get("/dict/list").json()["words"]
    for w in initial:
        client.post("/dict/remove", json={"word": w})

    r = client.post("/dict/add", json={"word": "ЦСНтест"})
    assert r.status_code == 200
    assert r.json()["added"] is True

    listing = client.get("/dict/list").json()
    assert "ЦСНтест" in listing["words"]

    r = client.post("/dict/remove", json={"word": "ЦСНтест"})
    assert r.status_code == 200
    assert r.json()["removed"] is True
    listing = client.get("/dict/list").json()
    assert "ЦСНтест" not in listing["words"]


def test_cloud_dict_disabled(cloud_module_dict_disabled):
    """/dict/list возвращает 503 если USER_DICT_ENABLED=false."""
    from fastapi.testclient import TestClient
    client = TestClient(cloud_module_dict_disabled.app)
    r = client.get("/dict/list")
    assert r.status_code == 503
    assert "отключён" in r.json()["error"]


def test_cloud_dict_add_invalid_input(cloud_module):
    """/dict/add возвращает 400 на невалидный JSON."""
    from fastapi.testclient import TestClient
    client = TestClient(cloud_module.app)
    r = client.post("/dict/add", json={})
    assert r.status_code == 400


def test_cloud_soft_linebreak_normalized(cloud_module, monkeypatch):
    """v2.2: одиночный \\r и \\r\\n на входе /suggest нормализуются до \\n\\n
    (paragraph), а одиночный \\n (Shift+Enter) — остаётся как мягкий перенос.

    Это исправляет баг «расширение разъединяет один абзац с Shift+Enter
    на несколько». Проверяем, что сервер передаёт модели текст в
    единой конвенции и raw_text в audit-логах нормализован.
    """
    from fastapi.testclient import TestClient

    captured: dict[str, str] = {}

    async def fake_call_model(messages, model):
        captured["user_msg"] = messages[-1]["content"]
        return (
            "===CORRECTED===\n"
            "Первая строка\nвторая строка.\n"
            "===CHANGES===\n"
            "1. Ошибок не найдено. Текст соответствует нормам.\n"
            "===END==="
        )

    monkeypatch.setattr(cloud_module, "call_model", fake_call_model)
    client = TestClient(cloud_module.app)
    # Имитируем то, что LibreOffice getString() мог бы прислать:
    # \r\n — paragraph break (Windows), \n — Shift+Enter (soft).
    raw = "Первая строка\nвторая строка.\r\nВторой абзац."
    files = {
        "text": ("t.txt", io.BytesIO(raw.encode("utf-8")), "text/plain"),
        "context": ("c.txt", io.BytesIO(b""), "text/plain"),
    }
    r = client.post("/suggest", files=files)
    assert r.status_code == 200
    sent = captured["user_msg"]
    # Soft-break (одиночный \n) сохранён внутри абзаца
    assert "Первая строка\nвторая строка." in sent
    # Hard-break (\r\n) превратился в paragraph break (\n\n)
    assert "Второй абзац." in sent
    assert "\r\n" not in sent
    assert "\r" not in sent


def test_cloud_normalize_line_breaks_helper(cloud_module):
    """Unit-тест нормализатора переносов: \\r\\n → \\n\\n, \\r → \\n\\n, \\n остаётся."""
    fn = cloud_module._normalize_line_breaks
    assert fn("a\r\nb") == "a\n\nb"
    assert fn("a\rb") == "a\n\nb"
    assert fn("a\nb") == "a\nb"  # soft break не трогаем
    assert fn("a\u2028b") == "a\nb"  # U+2028 — single soft break
    # Множественные \n\n схлопываются до 2
    assert fn("a\n\n\n\nb") == "a\n\nb"
    assert fn("") == ""


def test_cloud_postprocess_drops_eyo_substitutions(cloud_module, monkeypatch):
    """v2.2-revisited: cloud-сервер должен сбрасывать ё↔е стилистические
    правки в CHANGES и откатывать ё в CORRECTED по raw_text. SYSTEM_PROMPT
    уже запрещает такие правки, но safe-filter закрывает lost-in-the-middle
    bypass.
    """
    from fastapi.testclient import TestClient

    async def fake_call_model(messages, model):
        return (
            "===CORRECTED===\n"
            "проведёнными работами в третьем квартале\n"
            "===CHANGES===\n"
            "1. «проведенными» → «проведёнными» | расстановка буквы ё\n"
            "2. «третьем» → «третьем» | проверка идемпотентности\n"
            "===END==="
        )

    monkeypatch.setattr(cloud_module, "call_model", fake_call_model)
    client = TestClient(cloud_module.app)
    files = {
        "text": ("t.txt", io.BytesIO("проведенными работами в третьем квартале".encode("utf-8")), "text/plain"),
        "context": ("c.txt", io.BytesIO(b""), "text/plain"),
    }
    r = client.post("/suggest", files=files)
    assert r.status_code == 200
    # Пункт «проведенными → проведёнными» — ё-only подстановка, должна быть отфильтрована
    assert "«проведенными» → «проведёнными»" not in r.text
    # CORRECTED откатан к исходному написанию (ё → е по raw_text)
    assert "проведенными" in r.text


def test_cloud_postprocess_drops_not_in_text(cloud_module, monkeypatch):
    """Cloud-сервер дропает пункты, чей «было» отсутствует в raw_text
    (галлюцинации модели — как и local)."""
    from fastapi.testclient import TestClient

    async def fake_call_model(messages, model):
        return (
            "===CORRECTED===\n"
            "Документ согласно приказу.\n"
            "===CHANGES===\n"
            "1. «несуществующая_цитата» → «другое» | вымышленная правка\n"
            "===END==="
        )

    monkeypatch.setattr(cloud_module, "call_model", fake_call_model)
    client = TestClient(cloud_module.app)
    files = {
        "text": ("t.txt", io.BytesIO("Документ согласно приказу.".encode("utf-8")), "text/plain"),
        "context": ("c.txt", io.BytesIO(b""), "text/plain"),
    }
    r = client.post("/suggest", files=files)
    assert r.status_code == 200
    assert "несуществующая_цитата" not in r.text


def test_cloud_suggest_appends_end_marker(cloud_module, monkeypatch):
    """Если модель забыла ===END===, cloud-сервер дописывает его."""
    from fastapi.testclient import TestClient

    async def fake_call_model(messages, model):
        return (
            "===CORRECTED===\nок\n"
            "===CHANGES===\n1. Ошибок нет.\n"
        )

    monkeypatch.setattr(cloud_module, "call_model", fake_call_model)
    client = TestClient(cloud_module.app)
    files = {
        "text": ("t.txt", io.BytesIO("ок".encode("utf-8")), "text/plain"),
        "context": ("c.txt", io.BytesIO(b""), "text/plain"),
    }
    r = client.post("/suggest", files=files)
    assert r.status_code == 200
    assert "===END===" in r.text


def test_cloud_suggest_fallback_on_429(cloud_module, monkeypatch):
    """Cloud-сервер пробует следующую модель на HTTP 429."""
    from fastapi.testclient import TestClient
    import httpx as _httpx

    calls = []

    async def fake_call_model(messages, model):
        calls.append(model)
        if model == cloud_module.MODELS[0]:
            req = _httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
            resp = _httpx.Response(429, request=req, text='{"error":"rate-limited"}')
            raise _httpx.HTTPStatusError("rate-limited", request=req, response=resp)
        return (
            "===CORRECTED===\nок\n===CHANGES===\n1. Ошибок нет.\n===END==="
        )

    monkeypatch.setattr(cloud_module, "call_model", fake_call_model)
    client = TestClient(cloud_module.app)
    files = {
        "text": ("t.txt", io.BytesIO("ок".encode("utf-8")), "text/plain"),
        "context": ("c.txt", io.BytesIO(b""), "text/plain"),
    }
    r = client.post("/suggest", files=files)
    assert r.status_code == 200
    assert "===CORRECTED===" in r.text
    # Минимум 2 модели опробованы (429 на первой, успех на второй)
    assert len(calls) >= 2


def test_cloud_strip_thinking(cloud_module, monkeypatch):
    """Cloud-сервер срезает <think>…</think> блоки (parity с local)."""
    from fastapi.testclient import TestClient

    async def fake_call_model(messages, model):
        return (
            "<think>Думаю про орфографию...</think>\n"
            "===CORRECTED===\nок\n===CHANGES===\n1. Ошибок нет.\n===END==="
        )

    monkeypatch.setattr(cloud_module, "call_model", fake_call_model)
    client = TestClient(cloud_module.app)
    files = {
        "text": ("t.txt", io.BytesIO("ок".encode("utf-8")), "text/plain"),
        "context": ("c.txt", io.BytesIO(b""), "text/plain"),
    }
    r = client.post("/suggest", files=files)
    assert r.status_code == 200
    assert "<think>" not in r.text
    assert "Думаю про орфографию" not in r.text


def test_cloud_openrouter_client_resolve_models():
    """Резолвер моделей: preset A/B + override."""
    from shared.openrouter_client import resolve_models, CLOUD_PRESETS

    a = resolve_models("A")
    assert a[0] == "openrouter/free"
    b = resolve_models("B")
    assert b[0].startswith("qwen/qwen3-next")
    unknown = resolve_models("Z")  # неизвестный preset → A
    assert unknown == a
    override = resolve_models("A", override_models=["model-1", "model-2"])
    assert override == ["model-1", "model-2"]
    # Все 4 preset'а определены
    for k in ("A", "B", "C", "D"):
        assert k in CLOUD_PRESETS


def test_cloud_openrouter_client_key_redaction():
    """OpenRouterClient.key_redacted маскирует ключ."""
    from shared.openrouter_client import OpenRouterClient

    c1 = OpenRouterClient("sk-or-v1-abcdefghijklmnop1234")
    redacted = c1.key_redacted
    assert "sk-or-v1-abc" in redacted
    assert "1234" in redacted
    assert "..." in redacted
    assert "defghijkl" not in redacted

    c2 = OpenRouterClient("")
    assert c2.key_redacted == "(empty)"
    assert c2.key_present is False

    c3 = OpenRouterClient("ваш_ключ_тут")
    assert c3.key_present is False  # placeholder


# ─── v2.2.2: openrouter/free content=None и all-429 диагностика ───────


def _openrouter_mock_client(handler):
    """Создаёт OpenRouterClient с MockTransport для unit-тестов клиента."""
    import httpx
    from shared.openrouter_client import OpenRouterClient

    return OpenRouterClient(
        "sk-or-v1-test-token-do-not-use",
        transport=httpx.MockTransport(handler),
    )


def test_openrouter_post_chat_content_none_raises_softfail():
    """Регрессия: openrouter/free auto-router иногда отдаёт 200 OK
    + content=None. До v2.2.2 это валилось AttributeError 'NoneType'
    .strip(). Теперь должен подняться OpenRouterError (soft-fail),
    чтобы fallback ушёл на следующую модель."""
    import asyncio
    import httpx
    from shared.openrouter_client import OpenRouterError

    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": None}}],
        })

    client = _openrouter_mock_client(handler)
    with pytest.raises(OpenRouterError) as excinfo:
        asyncio.run(client._post_chat(
            [{"role": "user", "content": "hi"}],
            "openrouter/free",
            temperature=0.0, max_tokens=32,
        ))
    assert "content=None" in str(excinfo.value)
    # AttributeError больше не утекает
    assert "NoneType" not in str(excinfo.value)


def test_openrouter_post_chat_error_in_body_raises_softfail():
    """200 OK + error в теле (рейтлимит провайдера, content-policy) — soft-fail."""
    import asyncio
    import httpx
    from shared.openrouter_client import OpenRouterError

    def handler(request):
        return httpx.Response(200, json={
            "error": {"message": "rate limited by upstream", "code": 429},
        })

    client = _openrouter_mock_client(handler)
    with pytest.raises(OpenRouterError) as excinfo:
        asyncio.run(client._post_chat(
            [{"role": "user", "content": "hi"}],
            "openrouter/free",
            temperature=0.0, max_tokens=32,
        ))
    assert "error" in str(excinfo.value).lower()
    assert "rate limited" in str(excinfo.value)


def test_openrouter_post_chat_empty_choices_raises_softfail():
    """200 OK + пустой choices — soft-fail."""
    import asyncio
    import httpx
    from shared.openrouter_client import OpenRouterError

    def handler(request):
        return httpx.Response(200, json={"choices": []})

    client = _openrouter_mock_client(handler)
    with pytest.raises(OpenRouterError):
        asyncio.run(client._post_chat(
            [{"role": "user", "content": "hi"}],
            "model-x",
            temperature=0.0, max_tokens=32,
        ))


def test_openrouter_chat_all_429_returns_friendly_quota_message():
    """Когда ВСЕ модели возвращают HTTP 429, должно подняться
    OpenRouterError с понятным сообщением про исчерпанную квоту."""
    import asyncio
    import httpx
    from shared.openrouter_client import OpenRouterError

    def handler(request):
        return httpx.Response(429, json={"error": {"message": "Too Many Requests"}})

    client = _openrouter_mock_client(handler)
    with pytest.raises(OpenRouterError) as excinfo:
        asyncio.run(client.chat(
            [{"role": "user", "content": "hi"}],
            ["m1:free", "m2:free", "m3:free"],
            temperature=0.0, max_tokens=32,
        ))
    msg = str(excinfo.value)
    assert "429" in msg or "исчерпан" in msg
    assert "квот" in msg
    assert "OPENROUTER_MODELS" in msg


def test_openrouter_chat_fallback_skips_content_none_to_next():
    """Если первая модель отдала content=None, переходим к следующей,
    которая возвращает нормальный ответ."""
    import asyncio
    import httpx

    state = {"calls": 0}

    def handler(request):
        state["calls"] += 1
        if state["calls"] == 1:
            return httpx.Response(200, json={"choices": [{"message": {"content": None}}]})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "OK"}}],
        })

    client = _openrouter_mock_client(handler)
    content, used = asyncio.run(client.chat(
        [{"role": "user", "content": "hi"}],
        ["openrouter/free", "qwen/qwen3-next-80b-a3b-instruct:free"],
        temperature=0.0, max_tokens=32,
    ))
    assert content == "OK"
    assert used == "qwen/qwen3-next-80b-a3b-instruct:free"
    assert state["calls"] == 2


def test_cloud_chat_with_fallback_quota_exhausted_friendly_error(cloud_module, monkeypatch):
    """`_chat_with_fallback` в cloud/main.py отдаёт понятную ошибку,
    когда все MODELS вернули 429."""
    import asyncio
    import httpx
    from shared.openrouter_client import OpenRouterError

    async def fake_call_model(messages, model):
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        response = httpx.Response(429, request=request, json={
            "error": {"message": "Rate limit exceeded"},
        })
        raise httpx.HTTPStatusError("429", request=request, response=response)

    monkeypatch.setattr(cloud_module, "call_model", fake_call_model)
    monkeypatch.setattr(cloud_module, "MODELS", ["m1:free", "m2:free", "m3:free"])
    with pytest.raises(OpenRouterError) as excinfo:
        asyncio.run(cloud_module._chat_with_fallback([{"role": "user", "content": "hi"}]))
    msg = str(excinfo.value)
    assert "429" in msg
    assert "квот" in msg
    assert "OPENROUTER_MODELS" in msg
