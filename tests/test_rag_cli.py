"""Тесты CLI-команд `preview` и `stats` (без обращения к Ollama)."""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from shared.rag_cli import build_parser, cmd_preview, cmd_stats
from shared.rag_store import HashingEmbedder, RagStore


_GARANT_SAMPLE = """\
Система ГАРАНТ
Документ предоставлен КонсультантПлюс

Постановление Правительства РФ от 01.07.2025 № 100

1. Утвердить Положение о порядке согласования межведомственных проектов
2. Контроль за исполнением возложить на руководителей структурных подразделений.

Дата печати: 30.04.2026
Страница 1 из 12

© Гарант, 2024
"""


@pytest.fixture
def sample_garant_txt(tmp_path: Path) -> Path:
    p = tmp_path / "postanovlenie.txt"
    p.write_text(_GARANT_SAMPLE, encoding="utf-8")
    return p


def _parse(*argv) -> object:
    """Вспомогалка: распарсить CLI-аргументы."""
    parser = build_parser()
    return parser.parse_args(list(argv))


def _capture(func, args) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = func(args)
    assert rc == 0, f"command failed with code {rc}"
    return buf.getvalue()


# ─── preview ──────────────────────────────────────────────────
def test_preview_emits_json_summary_and_strips_garant(sample_garant_txt: Path):
    args = _parse("preview", str(sample_garant_txt), "--head", "0", "--json")
    out = _capture(cmd_preview, args)
    payload = json.loads(out)
    # Заголовки/копирайты Гаранта должны быть вычищены.
    head = payload["head"]
    assert "Система ГАРАНТ" not in head
    assert "© Гарант" not in head
    assert "Постановление Правительства" in head
    # Метрики очистки выглядят разумно.
    assert payload["text_len"] > 0
    assert payload["chunks"] >= 1
    assert payload["cleanup"]["original_lines"] > payload["cleanup"]["kept_lines"]


def test_preview_missing_file_returns_error_code(tmp_path: Path):
    args = _parse("preview", str(tmp_path / "nonexistent.txt"))
    rc = cmd_preview(args)
    assert rc == 2


def test_preview_text_mode_shows_head(sample_garant_txt: Path):
    args = _parse("preview", str(sample_garant_txt), "--head", "300")
    out = _capture(cmd_preview, args)
    assert "── Превью очищенного текста ──" in out
    assert "Постановление Правительства" in out


# ─── stats ───────────────────────────────────────────────────
def test_stats_empty_store(tmp_path: Path):
    args = _parse("--store-dir", str(tmp_path / "empty_store"), "stats")
    out = _capture(cmd_stats, args)
    assert "Хранилище пусто" in out


def test_stats_after_ingest(tmp_path: Path, sample_garant_txt: Path):
    store_dir = tmp_path / "store"
    store = RagStore(store_dir)
    store.add_document(
        doc_id="postanovlenie-100",
        file_path=sample_garant_txt,
        embedder=HashingEmbedder(dim=128),
        version="2025-07",
    )
    args = _parse("--store-dir", str(store_dir), "stats", "--json")
    out = _capture(cmd_stats, args)
    payload = json.loads(out)
    assert payload["documents"] == 1
    assert payload["chunks"] >= 1
    assert payload["vec_dim"] == 128
    assert any(name.startswith("hashing") for name in payload["embedders"])
    assert "2025-07" in payload["versions_top5"]
