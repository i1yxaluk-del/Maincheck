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
