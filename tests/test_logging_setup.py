"""Тесты настройки логгера."""
import logging
import os
from pathlib import Path

from shared.logging_setup import setup_logger


def test_creates_file_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_RETENTION_DAYS", "7")

    log1 = setup_logger("test_al")
    log1.info("hello")
    log2 = setup_logger("test_al")  # повторный вызов не должен дублировать handler'ы
    log2.warning("warn")
    assert log1 is log2
    # Файл создан
    assert (tmp_path / "test_al.log").exists()
    # Handler'ы не задваиваются
    assert len(log1.handlers) == 2  # file + console


def test_sibling_loggers_propagate_to_configured_parent(tmp_path, monkeypatch):
    """`setup_logger("ai_suggester.local")` должен также сконфигурировать
    родительский `ai_suggester`, чтобы логи из `ai_suggester.gec_bank`
    и других соседей попадали в тот же файл через propagation.

    До фикса v1.6.1: прогресс build_index() (`ai_suggester.gec_bank`)
    терялся — ни в файле, ни в journalctl. Диагностика была возможна
    только по логам Ollama.
    """
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    # Изолируем от прошлых тестов: чистим умбрелла-флаг и handlers.
    parent = logging.getLogger("isolated_parent_for_test")
    parent.handlers.clear()
    if hasattr(parent, "_ai_configured_umbrella"):
        delattr(parent, "_ai_configured_umbrella")

    main_log = setup_logger("isolated_parent_for_test.local")
    sibling = logging.getLogger("isolated_parent_for_test.gec_bank")
    sibling.info("test-sibling-message-123")

    # Форсим flush всех handlers родителя
    for h in parent.handlers:
        h.flush()
    log_file = tmp_path / "isolated_parent_for_test.local.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "test-sibling-message-123" in content, (
        "соседний логгер должен писать в тот же файл через зонтичный parent"
    )
    # И не должно быть дубликатов от самого main_log
    main_log.info("test-main-message-456")
    for h in parent.handlers:
        h.flush()
    for h in main_log.handlers:
        h.flush()
    content2 = log_file.read_text(encoding="utf-8")
    # Ровно одно вхождение основного сообщения (propagate=False у main_log)
    assert content2.count("test-main-message-456") == 1
