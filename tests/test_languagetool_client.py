"""v2.0-b: тесты LanguageTool-RU клиента.

Без живого LT-сервера — все запросы мокаются через httpx.MockTransport,
который инжектируется в LanguageToolClient через параметр `transport`.
Это не патчит httpx глобально, что важно — иначе ломается
starlette.TestClient (который наследуется от httpx.Client).
"""
from __future__ import annotations

import httpx
import pytest

from shared.languagetool_client import (
    LanguageToolClient,
    LTMatch,
    _parse_csv_env,
    get_languagetool_client,
    reset_client,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_client()
    yield
    reset_client()


def _transport(handler):
    return httpx.MockTransport(handler)


def test_parse_csv_env_empty():
    assert _parse_csv_env(None) == frozenset()
    assert _parse_csv_env("") == frozenset()
    assert _parse_csv_env("   ") == frozenset()


def test_parse_csv_env_basic():
    assert _parse_csv_env("STYLE,TYPOGRAPHY") == frozenset({"STYLE", "TYPOGRAPHY"})
    assert _parse_csv_env(" style , TYPO ") == frozenset({"STYLE", "TYPO"})
    assert _parse_csv_env(",STYLE,,TYPO,") == frozenset({"STYLE", "TYPO"})


def test_ltmatch_to_change_line():
    m = LTMatch(
        offset=10, length=4, before="дефис", suggestion="тире",
        message="Используйте длинное тире",
        category_id="TYPOGRAPHY", rule_id="DASH_RULE",
    )
    line = m.to_change_line(1)
    assert line.startswith("1.")
    assert "«дефис»" in line
    assert "«тире»" in line
    assert "[TYPOGRAPHY]" in line
    assert "Используйте длинное тире" in line


def test_ltmatch_truncates_long_message():
    long_msg = "A" * 200
    m = LTMatch(
        offset=0, length=1, before="x", suggestion="y",
        message=long_msg, category_id="STYLE", rule_id="R",
    )
    line = m.to_change_line(1)
    assert "AAAAAAA" in line
    assert long_msg not in line


def test_languagetool_unavailable_returns_empty():
    """Если httpx-запрос падает — available становится False, check() == []."""
    def handler(request):
        raise httpx.ConnectError("server down")

    client = LanguageToolClient(
        url="http://lt-mock:8081", timeout=2.0, transport=_transport(handler),
    )
    assert client.available is False
    assert client.check("Текст с ошибками.") == []


def test_languagetool_available_then_check():
    """Сервер отвечает на /v2/languages и возвращает matches от /v2/check."""
    def handler(request):
        if request.url.path == "/v2/languages":
            return httpx.Response(200, json=[
                {"name": "Russian", "code": "ru", "longCode": "ru-RU"},
            ])
        if request.url.path == "/v2/check":
            return httpx.Response(200, json={
                "matches": [
                    {
                        "message": "Используйте длинное тире",
                        "offset": 5, "length": 1,
                        "replacements": [{"value": "—"}],
                        "rule": {
                            "id": "DASH_RULE",
                            "category": {"id": "TYPOGRAPHY", "name": "Типографика"},
                        },
                    },
                ],
            })
        return httpx.Response(404)

    client = LanguageToolClient(
        url="http://lt-mock:8081", transport=_transport(handler),
    )
    assert client.available is True
    matches = client.check("Тест - проверка.")
    assert len(matches) == 1
    m = matches[0]
    assert m.before == "-"
    assert m.suggestion == "—"
    assert m.category_id == "TYPOGRAPHY"
    assert m.rule_id == "DASH_RULE"


def test_languagetool_skips_matches_without_replacements():
    """Матчи без replacements (только подсказка без замены) пропускаются."""
    def handler(request):
        if request.url.path == "/v2/languages":
            return httpx.Response(200, json=[{"longCode": "ru-RU"}])
        return httpx.Response(200, json={
            "matches": [
                {"message": "info", "offset": 0, "length": 1, "replacements": []},
                {
                    "message": "real", "offset": 5, "length": 1,
                    "replacements": [{"value": "—"}],
                    "rule": {"id": "R", "category": {"id": "TYPOGRAPHY"}},
                },
            ],
        })

    client = LanguageToolClient(
        url="http://lt-mock:8081", transport=_transport(handler),
    )
    assert client.available is True
    matches = client.check("X - Y.")
    assert len(matches) == 1
    assert matches[0].suggestion == "—"


def test_languagetool_skips_matches_with_invalid_offset():
    """Матчи с length <= 0 пропускаются (защита от мусорного ответа)."""
    def handler(request):
        if request.url.path == "/v2/languages":
            return httpx.Response(200, json=[{"longCode": "ru-RU"}])
        return httpx.Response(200, json={
            "matches": [
                {
                    "message": "bad", "offset": 0, "length": 0,
                    "replacements": [{"value": "X"}],
                    "rule": {"id": "R", "category": {"id": "STYLE"}},
                },
                {
                    "message": "good", "offset": 0, "length": 5,
                    "replacements": [{"value": "OK"}],
                    "rule": {"id": "R", "category": {"id": "STYLE"}},
                },
            ],
        })

    client = LanguageToolClient(
        url="http://lt-mock:8081", transport=_transport(handler),
    )
    assert client.available is True
    matches = client.check("Hello world")
    assert len(matches) == 1
    assert matches[0].suggestion == "OK"


def test_languagetool_500_returns_empty():
    """LT вернул 500 → клиент логирует warning и возвращает []."""
    def handler(request):
        if request.url.path == "/v2/languages":
            return httpx.Response(200, json=[{"longCode": "ru-RU"}])
        return httpx.Response(500, text="internal error")

    client = LanguageToolClient(
        url="http://lt-mock:8081", transport=_transport(handler),
    )
    assert client.available is True
    assert client.check("test") == []


def test_languagetool_check_empty_text():
    """Пустой текст → []. Без обращения к серверу."""
    client = LanguageToolClient(url="http://localhost:8081")
    assert client.check("") == []
    assert client.check("   ") == []


def test_languagetool_passes_categories():
    """Параметры enabled_categories передаются в /v2/check."""
    captured = {}

    def handler(request):
        if request.url.path == "/v2/languages":
            return httpx.Response(200, json=[{"longCode": "ru-RU"}])
        if request.url.path == "/v2/check":
            captured["body"] = request.content.decode()
            return httpx.Response(200, json={"matches": []})
        return httpx.Response(404)

    client = LanguageToolClient(
        url="http://lt-mock:8081",
        enabled_categories=frozenset({"STYLE", "TYPOGRAPHY"}),
        disabled_rules=frozenset({"BAD_RULE"}),
        transport=_transport(handler),
    )
    assert client.available is True
    client.check("Тест.")
    body = captured.get("body", "")
    assert "enabledCategories=" in body
    assert "STYLE" in body and "TYPOGRAPHY" in body
    assert "enabledOnly=false" in body
    assert "disabledRules=BAD_RULE" in body
    assert "language=ru-RU" in body


def test_get_languagetool_client_singleton():
    """get_languagetool_client возвращает один и тот же объект."""
    c1 = get_languagetool_client(url="http://localhost:8081")
    c2 = get_languagetool_client(url="http://localhost:9999")
    assert c1 is c2


def test_reset_client_clears_singleton():
    c1 = get_languagetool_client(url="http://localhost:8081")
    reset_client()
    c2 = get_languagetool_client(url="http://localhost:8081")
    assert c1 is not c2
