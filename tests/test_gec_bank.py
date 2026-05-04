"""Тесты банка few-shot пар и сборки сообщений."""
import json
from pathlib import Path

import pytest

from shared.gec_bank import (
    GecBank,
    GecPair,
    build_few_shot_messages,
    format_example_as_messages,
)
from shared.rag_store import HashingEmbedder


@pytest.fixture
def sample_bank_file(tmp_path: Path) -> Path:
    p = tmp_path / "bank.jsonl"
    pairs = [
        {
            "wrong": "Согласно приказа был утверждён план.",
            "right": "Согласно приказу был утверждён план.",
            "rule": "Падежное управление",
            "definition": "Предлог «согласно» требует дательного падежа.",
            "section": "Grammar",
        },
        {
            "wrong": "Отчёт был направлен, руководителю департамента.",
            "right": "Отчёт был направлен руководителю департамента.",
            "rule": "Лишняя запятая перед дополнением",
            "definition": "Запятая не ставится между сказуемым и косвенным дополнением.",
            "section": "Punctuation",
        },
        {
            "wrong": "Решением комиссии было постановлено.",
            "right": "Решением комиссии, было постановлено.",
            "rule": "Запятая при обособленном обстоятельстве",
            "definition": "Обстоятельство образа действия выделяется запятыми.",
            "section": "Punctuation",
        },
        # Пропускаемая строка: пустое поле
        {"wrong": "", "right": "X"},
        # Пропускаемая строка: идемпотентная
        {"wrong": "ok", "right": "ok"},
    ]
    with p.open("w", encoding="utf-8") as f:
        for d in pairs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return p


def _make_bank(sample_bank_file: Path) -> GecBank:
    bank = GecBank(HashingEmbedder(dim=256))
    n = bank.load_jsonl(sample_bank_file)
    bank.build_index()
    # 3 валидных, 2 пропущены
    assert n == 3
    assert len(bank) == 3
    return bank


def test_load_jsonl_skips_empty_and_idempotent(sample_bank_file: Path) -> None:
    bank = _make_bank(sample_bank_file)
    assert bank.stats()["total_pairs"] == 3
    assert bank.stats()["indexed_pairs"] == 3


def test_load_jsonl_missing_file_is_warning(tmp_path: Path) -> None:
    bank = GecBank(HashingEmbedder(dim=64))
    # Не падаем, возвращаем 0
    assert bank.load_jsonl(tmp_path / "nope.jsonl") == 0
    assert len(bank) == 0


def test_search_returns_sorted_top_k(sample_bank_file: Path) -> None:
    bank = _make_bank(sample_bank_file)
    # Текст с падежной ошибкой — должен ранжировать «согласно приказа» выше
    hits = bank.search("Согласно распоряжения был выпущен приказ.", top_k=2)
    assert len(hits) == 2
    scores = [s for s, _ in hits]
    assert scores == sorted(scores, reverse=True)
    # Hashing-embedder по лексическому пересечению — должен быть падежный
    # пример в первых двух (делит слова «согласно», «был», «приказ»)
    rules = [p.rule for _, p in hits]
    assert "Падежное управление" in rules


def test_search_empty_bank_returns_empty_list() -> None:
    bank = GecBank(HashingEmbedder(dim=32))
    # Нет build_index, нет entries — empty list
    assert bank.search("любой запрос", top_k=5) == []


def test_format_example_as_messages_round_trip() -> None:
    pair = GecPair(
        wrong="Согласно приказа был утверждён план.",
        right="Согласно приказу был утверждён план.",
        rule="Падежное управление",
    )
    msgs = format_example_as_messages(pair)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert pair.wrong in msgs[0]["content"]
    assert msgs[1]["role"] == "assistant"
    assistant = msgs[1]["content"]
    # Формат ответа должен быть как у реального сервера
    assert "===CORRECTED===" in assistant
    assert "===CHANGES===" in assistant
    assert "===END===" in assistant
    assert pair.right in assistant
    assert pair.rule in assistant


def test_build_few_shot_messages_order() -> None:
    pairs = [
        GecPair(wrong="A1", right="A2", rule="r1"),
        GecPair(wrong="B1", right="B2", rule="r2"),
    ]
    msgs = build_few_shot_messages(
        system_prompt="SYS",
        user_text="USER",
        examples=pairs,
        extra_system="EXTRA",
    )
    # Порядок: system → extra_system → (user, assistant) × N → user
    assert [m["role"] for m in msgs] == [
        "system", "system",
        "user", "assistant",
        "user", "assistant",
        "user",
    ]
    assert msgs[0]["content"] == "SYS"
    assert msgs[1]["content"] == "EXTRA"
    # Последний user — реальный текст
    assert msgs[-1]["content"] == "USER"
    # Содержимое примеров в порядке передачи
    assert "A1" in msgs[2]["content"]
    assert "A2" in msgs[3]["content"]
    assert "B1" in msgs[4]["content"]
    assert "B2" in msgs[5]["content"]


def test_build_few_shot_messages_no_extra_system() -> None:
    msgs = build_few_shot_messages(
        system_prompt="SYS",
        user_text="USER",
        examples=[],
    )
    assert [m["role"] for m in msgs] == ["system", "user"]


def test_seed_bank_file_is_valid() -> None:
    """Проверяет, что встроенный seed-банк парсится без ошибок."""
    here = Path(__file__).resolve().parent
    seed = here.parent / "Сервер" / "shared" / "gec_seed" / "gec_bank.jsonl"
    assert seed.exists(), f"seed-банк отсутствует: {seed}"
    bank = GecBank(HashingEmbedder(dim=64))
    n = bank.load_jsonl(seed)
    # Ожидаем минимум 100 пар (LORuGEC дал 288, но даём запас на фильтрацию)
    assert n >= 100, f"seed-банк слишком маленький: {n} пар"
    bank.build_index()
    # Проверяем основные поля
    sample = bank.search("запятая при обособлении", top_k=3)
    assert len(sample) == 3
    for score, pair in sample:
        assert pair.wrong
        assert pair.right
        assert pair.wrong != pair.right
