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
