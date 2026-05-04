"""
Настройка логирования для серверов AI Suggester.

Формат: timestamp · level · logger · message
Ротация: сутки, retention по переменной окружения LOG_RETENTION_DAYS
(0 → хранить бесконечно).
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path


_LOG_FORMAT = "%(asctime)s · %(levelname)-7s · %(name)s · %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(name: str = "ai_suggester") -> logging.Logger:
    """Инициализирует и возвращает именованный логгер.

    Переменные окружения:
        LOG_LEVEL          — DEBUG/INFO/WARNING/ERROR (по умолчанию INFO)
        LOG_DIR            — каталог для файлов логов (по умолчанию logs/)
        LOG_RETENTION_DAYS — 0 = без ограничений, иначе число дней хранения
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    try:
        retention_days = int(os.getenv("LOG_RETENTION_DAYS", "30"))
    except ValueError:
        retention_days = 30

    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Не дублировать при повторном импорте
    if getattr(logger, "_ai_configured", False):
        return logger

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Файл с посуточной ротацией
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_dir / f"{name}.log",
        when="midnight",
        interval=1,
        backupCount=retention_days if retention_days > 0 else 0,
        encoding="utf-8",
        utc=False,
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    # Консоль (stderr — чтобы не ломать PlainTextResponse)
    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(formatter)
    console.setLevel(level)
    logger.addHandler(console)

    logger.propagate = False
    logger._ai_configured = True  # type: ignore[attr-defined]

    # Также конфигурируем «зонтичный» родительский логгер теми же handlers —
    # чтобы соседние модули (`ai_suggester.gec_bank`, `ai_suggester.rag`,
    # `ai_suggester.audit`) писали в тот же файл через propagation. Без
    # этого их логи молча теряются (только WARNING утекают через root
    # lastResort на stderr). Это и скрывало прогресс `GecBank.build_index()`
    # в журнале v1.6.0.
    if "." in name:
        parent_name = name.rsplit(".", 1)[0]
        parent = logging.getLogger(parent_name)
        if not getattr(parent, "_ai_configured_umbrella", False):
            parent.setLevel(level)
            parent.addHandler(file_handler)
            parent.addHandler(console)
            # parent.propagate остаётся True по умолчанию, но наверху
            # (root) обычно нет AI-handlers, поэтому дубликатов нет.
            parent._ai_configured_umbrella = True  # type: ignore[attr-defined]

    logger.info(
        "Логгер инициализирован (level=%s, dir=%s, retention=%sд)",
        level_name, log_dir, retention_days,
    )
    return logger
