"""Tests for v1.8b user_dict — пользовательский словарь.

Покрывает:
  * load/save/atomic-write
  * валидацию входа (форбидден chars, длина, лимит)
  * add/remove/contains
  * render_for_prompt (инжекция в SYSTEM_PROMPT)
  * thread-safety (через ThreadPoolExecutor)
"""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

# Подключаем shared/ к sys.path как в основном коде
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "server"))

from shared.user_dict import (  # noqa: E402
    MAX_WORDS,
    MAX_WORD_LEN,
    MIN_WORD_LEN,
    UserDictError,
    UserDictionary,
    _validate_word,
    reset_user_dict_for_tests,
)


@pytest.fixture
def tmp_dict_path(tmp_path: Path) -> Path:
    return tmp_path / "user_dict.json"


@pytest.fixture
def empty_dict(tmp_dict_path: Path) -> UserDictionary:
    return UserDictionary(tmp_dict_path)


# ─── Validation ──────────────────────────────────────────────────────


def test_validate_word_strips_whitespace():
    assert _validate_word("  ЦСН  ") == "ЦСН"


def test_validate_word_rejects_empty():
    with pytest.raises(UserDictError, match="пустая"):
        _validate_word("")
    with pytest.raises(UserDictError, match="пустая"):
        _validate_word("   ")


def test_validate_word_rejects_too_short():
    with pytest.raises(UserDictError, match="длиннее"):
        _validate_word("А")  # 1 буква


def test_validate_word_rejects_too_long():
    with pytest.raises(UserDictError, match="не должно быть длиннее"):
        _validate_word("А" * (MAX_WORD_LEN + 1))


def test_validate_word_rejects_forbidden_chars():
    with pytest.raises(UserDictError, match="запрещённые"):
        _validate_word("ЦС\nН")
    with pytest.raises(UserDictError, match="запрещённые"):
        _validate_word("ЦСН<script>")
    with pytest.raises(UserDictError, match="запрещённые"):
        _validate_word('ЦСН"')


def test_validate_word_rejects_spaces():
    with pytest.raises(UserDictError, match="пробел"):
        _validate_word("Центр Специального Назначения")


def test_validate_word_accepts_dash_and_digits():
    assert _validate_word("КС-2") == "КС-2"
    assert _validate_word("ГОСТ-12345") == "ГОСТ-12345"


def test_validate_word_rejects_non_string():
    with pytest.raises(UserDictError, match="строкой"):
        _validate_word(123)  # type: ignore[arg-type]


# ─── Basic add/remove/list ───────────────────────────────────────────


def test_add_returns_true_on_new(empty_dict: UserDictionary):
    assert empty_dict.add("ЦСН") is True
    assert empty_dict.list_words() == ["ЦСН"]


def test_add_returns_false_on_duplicate(empty_dict: UserDictionary):
    empty_dict.add("ЦСН")
    assert empty_dict.add("ЦСН") is False
    assert empty_dict.list_words() == ["ЦСН"]


def test_remove_returns_true_on_existing(empty_dict: UserDictionary):
    empty_dict.add("ЦСН")
    assert empty_dict.remove("ЦСН") is True
    assert empty_dict.list_words() == []


def test_remove_returns_false_on_missing(empty_dict: UserDictionary):
    assert empty_dict.remove("ЦСН") is False


def test_contains_case_insensitive(empty_dict: UserDictionary):
    empty_dict.add("ЦСН")
    assert empty_dict.contains("ЦСН")
    assert empty_dict.contains("цсн")
    assert empty_dict.contains("Цсн")
    assert not empty_dict.contains("ЦНС")


def test_list_words_sorted(empty_dict: UserDictionary):
    empty_dict.add("ЦСН")
    empty_dict.add("МЧС")
    empty_dict.add("УФ")
    assert empty_dict.list_words() == ["МЧС", "УФ", "ЦСН"]


def test_as_frozenset_immutable(empty_dict: UserDictionary):
    empty_dict.add("ЦСН")
    empty_dict.add("МЧС")
    fs = empty_dict.as_frozenset()
    assert fs == frozenset({"ЦСН", "МЧС"})
    # Изменение dict не аффектит уже полученный snapshot
    empty_dict.add("УФ")
    assert "УФ" not in fs


# ─── Persistence ─────────────────────────────────────────────────────


def test_save_and_reload(tmp_dict_path: Path):
    d1 = UserDictionary(tmp_dict_path)
    d1.add("ЦСН")
    d1.add("КС-2")
    # Проверяем что файл записан
    assert tmp_dict_path.exists()
    data = json.loads(tmp_dict_path.read_text(encoding="utf-8"))
    assert "words" in data
    assert "updated_at" in data
    assert sorted(data["words"]) == ["КС-2", "ЦСН"]

    # Новый инстанс читает то же
    d2 = UserDictionary(tmp_dict_path)
    assert d2.list_words() == ["КС-2", "ЦСН"]


def test_load_corrupt_file_starts_empty(tmp_dict_path: Path):
    tmp_dict_path.write_text("not valid json {{{", encoding="utf-8")
    d = UserDictionary(tmp_dict_path)
    assert d.list_words() == []


def test_load_missing_file_starts_empty(tmp_dict_path: Path):
    assert not tmp_dict_path.exists()
    d = UserDictionary(tmp_dict_path)
    assert d.list_words() == []


def test_atomic_write_preserves_old_on_save_fail(tmp_dict_path: Path, monkeypatch):
    """При падении os.replace старый файл не должен исчезнуть."""
    d = UserDictionary(tmp_dict_path)
    d.add("ЦСН")
    original = tmp_dict_path.read_text(encoding="utf-8")

    # Симулируем сбой записи
    import os as _os

    def boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(_os, "replace", boom)
    with pytest.raises(UserDictError, match="не удалось"):
        d.add("МЧС")
    # Файл всё ещё на месте
    assert tmp_dict_path.read_text(encoding="utf-8") == original
    # In-memory тоже rollback
    assert "МЧС" not in d.list_words()


# ─── Limits ──────────────────────────────────────────────────────────


def test_max_words_limit(tmp_dict_path: Path):
    d = UserDictionary(tmp_dict_path)
    for i in range(MAX_WORDS):
        d.add(f"СЛОВО{i:03d}")
    assert len(d.list_words()) == MAX_WORDS
    with pytest.raises(UserDictError, match="лимит"):
        d.add("ОВЕРФЛОУ")


# ─── Prompt rendering ────────────────────────────────────────────────


def test_render_for_prompt_empty(empty_dict: UserDictionary):
    assert empty_dict.render_for_prompt() == ""


def test_render_for_prompt_non_empty(empty_dict: UserDictionary):
    empty_dict.add("ЦСН")
    empty_dict.add("КС-2")
    rendered = empty_dict.render_for_prompt()
    assert "корректн" in rendered.lower()
    assert "ЦСН" in rendered
    assert "КС-2" in rendered


# ─── Thread safety ───────────────────────────────────────────────────


def test_concurrent_add(tmp_dict_path: Path):
    """100 потоков одновременно добавляют разные слова — ничего не теряется."""
    d = UserDictionary(tmp_dict_path)

    def add_one(i: int) -> bool:
        return d.add(f"СЛОВО{i:03d}")

    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(add_one, range(100)))
    assert sum(results) == 100
    assert len(d.list_words()) == 100


# ─── Singleton ───────────────────────────────────────────────────────


def test_reset_singleton(tmp_dict_path: Path):
    """Singleton helper для тестов корректно сбрасывает кэшированный инстанс."""
    from shared.user_dict import get_user_dict
    reset_user_dict_for_tests()
    d1 = get_user_dict(tmp_dict_path)
    d1.add("ЦСН")
    d2 = get_user_dict()  # default path не передаём — должен вернуть тот же singleton
    assert d2 is d1
    reset_user_dict_for_tests()
