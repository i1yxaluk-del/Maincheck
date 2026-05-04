"""
Аудит запросов: SQLite-журнал «кто что спрашивал и что получил».

Поля:
    id, ts (UTC ISO), client_ip, user_agent, server,
    model, text_len, context_len, changes_count,
    duration_ms, ok, error, text_sha1, text_snippet

Параметры из окружения:
    AUDIT_ENABLED      — true/false
    AUDIT_DB           — путь к .sqlite (по умолчанию logs/audit.sqlite)
    AUDIT_REDACT_TEXT  — если true, поле text_snippet пустое
    LOG_RETENTION_DAYS — purge старше N дней при каждом запуске
"""
from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT NOT NULL,
    client_ip      TEXT,
    user_agent     TEXT,
    server         TEXT,
    model          TEXT,
    text_len       INTEGER,
    context_len    INTEGER,
    changes_count  INTEGER,
    duration_ms    INTEGER,
    ok             INTEGER,
    error          TEXT,
    text_sha1      TEXT,
    text_snippet   TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit(ts);
"""

_log = logging.getLogger("ai_suggester.audit")


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class AuditStore:
    """Тонкая обёртка над SQLite с sha1-дедупликацией и авто-purge."""

    def __init__(self, db_path: Optional[str] = None, *, enabled: Optional[bool] = None,
                 redact: Optional[bool] = None, retention_days: Optional[int] = None):
        self.enabled = _parse_bool(os.getenv("AUDIT_ENABLED", "true")) if enabled is None else enabled
        self.redact = _parse_bool(os.getenv("AUDIT_REDACT_TEXT", "false")) if redact is None else redact
        try:
            self.retention_days = int(os.getenv("LOG_RETENTION_DAYS", "30")) if retention_days is None else retention_days
        except ValueError:
            self.retention_days = 30

        self.db_path = Path(db_path or os.getenv("AUDIT_DB", "logs/audit.sqlite"))
        self._lock = threading.Lock()

        if self.enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._conn() as c:
                c.executescript(_SCHEMA)
            self.purge_old()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def purge_old(self) -> int:
        """Удалить записи старше retention_days. Возвращает число удалённых."""
        if not self.enabled or self.retention_days <= 0:
            return 0
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.retention_days)).isoformat()
        with self._lock, self._conn() as c:
            cur = c.execute("DELETE FROM audit WHERE ts < ?", (cutoff,))
            deleted = cur.rowcount or 0
        if deleted:
            _log.info("Audit purge: удалено %d записей старше %d дней", deleted, self.retention_days)
        return deleted

    def record(
        self,
        *,
        client_ip: str = "",
        user_agent: str = "",
        server: str = "",
        model: str = "",
        text: str = "",
        context: str = "",
        changes_count: int = 0,
        duration_ms: int = 0,
        ok: bool = True,
        error: str = "",
    ) -> None:
        if not self.enabled:
            return
        snippet = "" if self.redact else text[:200]
        sha1 = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest() if text else ""
        ts = datetime.now(timezone.utc).isoformat()
        try:
            with self._lock, self._conn() as c:
                c.execute(
                    """INSERT INTO audit
                       (ts, client_ip, user_agent, server, model,
                        text_len, context_len, changes_count,
                        duration_ms, ok, error, text_sha1, text_snippet)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (ts, client_ip, user_agent, server, model,
                     len(text), len(context), int(changes_count),
                     int(duration_ms), 1 if ok else 0, error[:500], sha1, snippet),
                )
        except Exception as e:  # pragma: no cover
            _log.warning("Не удалось записать в аудит: %s", e)

    def stats(self, hours: int = 24) -> dict:
        """Сводка за последние N часов — для эндпоинта /metrics."""
        if not self.enabled:
            return {"enabled": False}
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with self._lock, self._conn() as c:
            row = c.execute(
                """SELECT
                     COUNT(*) AS total,
                     SUM(ok) AS ok,
                     AVG(duration_ms) AS avg_ms,
                     AVG(changes_count) AS avg_changes
                   FROM audit WHERE ts >= ?""",
                (cutoff,),
            ).fetchone()
        total, ok, avg_ms, avg_changes = row
        return {
            "enabled": True,
            "window_hours": hours,
            "total": total or 0,
            "ok": ok or 0,
            "fail": (total or 0) - (ok or 0),
            "avg_duration_ms": round(avg_ms or 0.0, 1),
            "avg_changes": round(avg_changes or 0.0, 2),
        }


class Timer:
    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.ms = int((time.perf_counter() - self._t) * 1000)


def count_changes(response_text: str) -> int:
    """Считает пункты в блоке ===CHANGES=== … ===END=== ответа модели."""
    if "===CHANGES===" not in response_text:
        return 0
    try:
        section = response_text.split("===CHANGES===", 1)[1]
        section = section.split("===END===", 1)[0]
    except IndexError:
        return 0
    count = 0
    for line in section.splitlines():
        line = line.strip()
        if not line or line.startswith("==="):
            continue
        # нумерованный пункт вида "1. ..." или "№. ..."
        if line[:3].replace(".", "").strip().isdigit() or line.startswith(("№", "-", "•")):
            count += 1
    if count == 0 and "Ошибок не найдено" in section:
        return 0
    return count
