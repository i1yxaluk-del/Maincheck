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
    logger.info(
        "Логгер инициализирован (level=%s, dir=%s, retention=%sд)",
        level_name, log_dir, retention_days,
    )
    return logger
