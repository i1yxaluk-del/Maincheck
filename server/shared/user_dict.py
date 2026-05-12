"""Пользовательский словарь администратора (v1.8b).

Юзер из LibreOffice-расширения может добавлять туда:
  * аббревиатуры (ЦСН, КС-2, УФ, ВО)
  * собственные имена / служебные слова, незнакомые модели
  * локальные термины (например, названия отделов, должностей)

Серверный эффект:
  1. При каждом /suggest-запросе слова из словаря инжектируются в
     SYSTEM_PROMPT инструкцией «не считать ошибками следующие термины:
     ЦСН, КС-2, УФ, …».
  2. В morph_detector передаётся `whitelist` — слова исключаются из
     OOV-проверки (детектор не флагует их как «нет в словаре»).
  3. В пост-фильтре дропаются пункты CHANGES, в которых модель
     пытается «исправить» whitelisted-слово (before=ЦСН, after=ЦНС
     и т.п.).

Хранение:
  * data/user_dict.json — JSON-файл `{"words": [...], "updated_at": "..."}`
  * Atomic-write (через temp + rename), thread-safe через файл-блокировку
    через простую lock-семантику (одиночный uvicorn-worker, не нужен
    process-level lock).

API сервера:
  * GET  /dict/list  → {"words": [...]}
  * POST /dict/add   {"word": "ЦСН"} → 201
  * POST /dict/remove {"word": "ЦСН"} → 200
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_log = logging.getLogger("ai_suggester.user_dict")


# Не более 500 слов в словаре чтобы не раздуть SYSTEM_PROMPT и не
# выходить за num_ctx модели T-lite.
MAX_WORDS = 500
# Минимальная и максимальная длина слова (защита от мусора).
MIN_WORD_LEN = 2
MAX_WORD_LEN = 64
# Запрещённые символы — защита от prompt injection и от того что
# юзер случайно отправит абзац / новую строку как «слово».
_FORBIDDEN_CHARS = re.compile(r"[\x00-\x1f\n\r\t<>{}\\\"`]")


class UserDictError(ValueError):
    """Ошибка валидации входа (юзер прислал мусор / превышение лимита)."""


def _validate_word(word: str) -> str:
    """Проверяет и нормализует слово. Бросает UserDictError при некорректном вводе."""
    if not isinstance(word, str):
        raise UserDictError("слово должно быть строкой")
    w = word.strip()
    if not w:
        raise UserDictError("пустая строка не допускается")
    if len(w) < MIN_WORD_LEN:
        raise UserDictError(f"слово должно быть длиннее {MIN_WORD_LEN-1} символов")
    if len(w) > MAX_WORD_LEN:
        raise UserDictError(f"слово не должно быть длиннее {MAX_WORD_LEN} символов")
    if _FORBIDDEN_CHARS.search(w):
        raise UserDictError(
            "слово содержит запрещённые символы (управляющие, кавычки, скобки)"
        )
    # Не должно содержать пробелы или знаки препинания (это «слово», не фраза)
    if any(c.isspace() for c in w):
        raise UserDictError("слово не должно содержать пробелов")
    return w


class UserDictionary:
    """Словарь пользовательских терминов с persistent storage.

    Thread-safe в пределах одного uvicorn-worker'а (используется
    threading.Lock для атомарных операций и atomic-write для дисковой
    записи).
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._words: set[str] = set()
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> None:
        if not self._path.exists():
            self._words = set()
            _log.info(
                "UserDict: файл не найден (%s), начинаю с пустого словаря",
                self._path,
            )
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            words = data.get("words", []) if isinstance(data, dict) else []
            if not isinstance(words, list):
                words = []
            self._words = {
                _validate_word(w) for w in words if isinstance(w, str) and w.strip()
            }
            _log.info(
                "UserDict: загружено %d слов из %s", len(self._words), self._path
            )
        except (OSError, json.JSONDecodeError, UserDictError) as e:
            _log.warning(
                "UserDict: не удалось загрузить %s (%s) — начинаю с пустого",
                self._path,
                e,
            )
            self._words = set()

    def _save(self) -> None:
        """Atomic write: temp-файл + rename. На случай падения сервера
        в момент записи — старый файл не повреждается.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        data = {
            "words": sorted(self._words),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self._path)

    def list_words(self) -> list[str]:
        """Возвращает отсортированный список слов (snapshot)."""
        with self._lock:
            return sorted(self._words)

    def add(self, word: str) -> bool:
        """Добавляет слово. Возвращает True если добавлено (False — уже было)."""
        w = _validate_word(word)
        with self._lock:
            if w in self._words:
                return False
            if len(self._words) >= MAX_WORDS:
                raise UserDictError(
                    f"достигнут лимит словаря ({MAX_WORDS} слов); удалите старые перед добавлением"
                )
            self._words.add(w)
            try:
                self._save()
            except OSError as e:
                # Откатываем in-memory изменение если запись на диск не удалась.
                self._words.discard(w)
                raise UserDictError(f"не удалось сохранить словарь: {e}") from e
            _log.info("UserDict: добавлено слово %r (всего %d)", w, len(self._words))
            return True

    def remove(self, word: str) -> bool:
        """Удаляет слово. Возвращает True если удалено (False — не было)."""
        w = _validate_word(word)
        with self._lock:
            if w not in self._words:
                return False
            self._words.discard(w)
            try:
                self._save()
            except OSError as e:
                self._words.add(w)
                raise UserDictError(f"не удалось сохранить словарь: {e}") from e
            _log.info("UserDict: удалено слово %r (всего %d)", w, len(self._words))
            return True

    def contains(self, word: str) -> bool:
        """Case-insensitive проверка наличия слова в словаре."""
        with self._lock:
            wl = word.strip().lower()
            return any(w.lower() == wl for w in self._words)

    def as_frozenset(self) -> frozenset[str]:
        """Snapshot для передачи в morph_detector (immutable)."""
        with self._lock:
            return frozenset(self._words)

    def render_for_prompt(self) -> str:
        """Render-инструкция для инжекции в SYSTEM_PROMPT.

        Возвращает пустую строку если словарь пустой (тогда инструкцию
        не подмешиваем).
        """
        with self._lock:
            if not self._words:
                return ""
            sorted_words = sorted(self._words)
            joined = ", ".join(sorted_words)
            return (
                "Следующие термины являются корректными аббревиатурами/названиями "
                "и НЕ должны рассматриваться как ошибки: " + joined + "."
            )


# ─── Singleton ────────────────────────────────────────────────────────


_INSTANCE: Optional[UserDictionary] = None


def get_user_dict(path: Optional[Path] = None) -> UserDictionary:
    """Возвращает singleton UserDictionary.

    Путь по умолчанию: `<repo_root>/data/user_dict.json`. Можно
    переопределить через env `AI_SUGGESTER_USER_DICT_PATH` (тестовая VM).
    """
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE
    if path is None:
        env_path = os.environ.get("AI_SUGGESTER_USER_DICT_PATH")
        if env_path:
            path = Path(env_path)
        else:
            # repo_root/data/user_dict.json (server/local/main.py: ../../data)
            here = Path(__file__).resolve()
            repo_root = here.parents[2]
            path = repo_root / "data" / "user_dict.json"
    _INSTANCE = UserDictionary(path)
    return _INSTANCE


def reset_user_dict_for_tests() -> None:
    """Сбрасывает singleton — только для тестов."""
    global _INSTANCE
    _INSTANCE = None
