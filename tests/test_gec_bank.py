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


def test_load_jsonl_skips_pairs_with_nested_quotes(tmp_path: Path) -> None:
    """Пары с внешними кавычками в wrong/right ломают _CHANGE_PAIR_RE
    на сервере (обрезка цитаты до первой внутренней кавычки). Такие пары
    должны отбрасываться на этапе загрузки с warning'ом.
    """
    p = tmp_path / "bank_nested.jsonl"
    items = [
        # OK — без кавычек
        {"wrong": "согласно приказа", "right": "согласно приказу", "rule": "rA"},
        # Skip — «» внутри wrong
        {
            "wrong": 'В Wildberries рассказали «Ведомостям», что покупка.',
            "right": 'В Wildberries рассказали «Ведомостям» что покупка.',
            "rule": "rB",
        },
        # Skip — " внутри right
        {
            "wrong": "Он сказал что приедет",
            "right": 'Он сказал "что приедет"',
            "rule": "rC",
        },
        # Skip — „" (типографские)
        {
            "wrong": "текст, „внутри“, продолжение",
            "right": "текст „внутри“ продолжение",
            "rule": "rD",
        },
        # OK — апостроф ASCII (НЕ в _OUTER_QUOTE_CHARS)
        {"wrong": "I don't know", "right": "I don't know тут", "rule": "rE"},
    ]
    with p.open("w", encoding="utf-8") as f:
        for d in items:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    bank = GecBank(HashingEmbedder(dim=64))
    n = bank.load_jsonl(p)
    # Ожидаем: rA, rE — прошли; rB, rC, rD — отброшены
    assert n == 2, f"ожидалось 2 пары, получили {n}"
    assert [e.pair.rule for e in bank._entries] == ["rA", "rE"]


def test_seed_bank_no_nested_quotes_after_load() -> None:
    """После фильтра ни одна загруженная пара не должна содержать
    внешние кавычки. Это инвариант для безопасного few-shot."""
    from shared.gec_bank import _OUTER_QUOTE_CHARS

    here = Path(__file__).resolve().parent
    seed = here.parent / "server" / "shared" / "gec_seed" / "gec_bank.jsonl"
    bank = GecBank(HashingEmbedder(dim=64))
    n = bank.load_jsonl(seed)
    assert n > 0
    for entry in bank._entries:
        for c in _OUTER_QUOTE_CHARS:
            assert c not in entry.pair.wrong, (
                f"пара с {c!r} в wrong не должна попасть в банк: {entry.pair.wrong!r}"
            )
            assert c not in entry.pair.right, (
                f"пара с {c!r} в right не должна попасть в банк: {entry.pair.right!r}"
            )


def test_build_index_cache_roundtrip(tmp_path: Path, sample_bank_file: Path) -> None:
    """Первый build_index сохраняет кэш; второй поднимает векторы оттуда,
    минуя embedder (чтобы старт сервера не упирался в 278 × 8 с)."""

    class _CountingEmbedder:
        name = "mock:hash:8"

        def __init__(self) -> None:
            self.calls = 0

        @property
        def dim(self) -> int:
            return 8

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls += 1
            base = HashingEmbedder(dim=8)
            return base.embed(texts)

    cache = tmp_path / "index.pkl"
    e1 = _CountingEmbedder()
    bank1 = GecBank(e1)
    bank1.load_jsonl(sample_bank_file)
    bank1.build_index(cache_path=cache)
    assert e1.calls == 1, "первый build_index должен вызвать embedder ровно один раз"
    assert cache.exists(), "после первого build_index должен появиться файл кэша"

    # Второй bank — тот же эмбеддер по имени, тот же банк → должен грузить кэш
    e2 = _CountingEmbedder()
    bank2 = GecBank(e2)
    bank2.load_jsonl(sample_bank_file)
    bank2.build_index(cache_path=cache)
    assert e2.calls == 0, "при валидном кэше embedder вызываться не должен"
    assert bank2._indexed_count == bank1._indexed_count
    # Векторы из кэша эквивалентны исходным
    vecs1 = [list(e.vec) for e in bank1._entries]
    vecs2 = [list(e.vec) for e in bank2._entries]
    assert vecs1 == vecs2


def test_build_index_cache_invalidates_on_content_change(
    tmp_path: Path, sample_bank_file: Path
) -> None:
    """Если банк изменился (добавили пару) — fingerprint меняется, кэш
    игнорируется, embedder вызывается заново."""
    cache = tmp_path / "index.pkl"

    class _CountingEmbedder:
        name = "mock:hash:8"

        def __init__(self) -> None:
            self.calls = 0

        @property
        def dim(self) -> int:
            return 8

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls += 1
            return HashingEmbedder(dim=8).embed(texts)

    e1 = _CountingEmbedder()
    bank1 = GecBank(e1)
    bank1.load_jsonl(sample_bank_file)
    bank1.build_index(cache_path=cache)

    # Добавляем пару в jsonl
    with sample_bank_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "wrong": "Новая ошибка",
            "right": "Новая правка",
            "rule": "тест",
        }, ensure_ascii=False) + "\n")

    e2 = _CountingEmbedder()
    bank2 = GecBank(e2)
    bank2.load_jsonl(sample_bank_file)
    bank2.build_index(cache_path=cache)
    assert e2.calls == 1, "после изменения банка embedder должен быть вызван"


def test_build_index_cache_invalidates_on_embedder_change(
    tmp_path: Path, sample_bank_file: Path
) -> None:
    """Смена эмбеддера (hashing:64 → hashing:128) должна инвалидировать
    кэш — иначе размерности векторов в банке и в запросе не совпадут."""
    cache = tmp_path / "index.pkl"
    bank1 = GecBank(HashingEmbedder(dim=64))
    bank1.load_jsonl(sample_bank_file)
    bank1.build_index(cache_path=cache)
    dim1 = len(bank1._entries[0].vec)

    bank2 = GecBank(HashingEmbedder(dim=128))
    bank2.load_jsonl(sample_bank_file)
    bank2.build_index(cache_path=cache)
    dim2 = len(bank2._entries[0].vec)

    assert dim1 == 64 and dim2 == 128, (
        "кэш должен переиндексироваться при смене эмбеддера (dim)"
    )


def test_seed_bank_file_is_valid() -> None:
    """Проверяет, что встроенный seed-банк парсится без ошибок."""
    here = Path(__file__).resolve().parent
    seed = here.parent / "server" / "shared" / "gec_seed" / "gec_bank.jsonl"
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


def test_extended_bank_file_is_valid() -> None:
    """Расширенный банк (LORuGEC full, v1.6.2) тоже парсится и индексируется."""
    here = Path(__file__).resolve().parent
    seed = here.parent / "server" / "shared" / "gec_seed" / "gec_bank.jsonl"
    extended = here.parent / "server" / "shared" / "gec_seed" / "gec_bank_extended.jsonl"
    assert extended.exists(), f"расширенный банк отсутствует: {extended}"
    bank = GecBank(HashingEmbedder(dim=64))
    # Имитируем прод-конфиг: подгружаем оба файла из GEC_BANK_FILES.
    n = bank.load_jsonl(seed, extended)
    # Объединённый банк ≥ 800 пар (288 базовых + 639 LORuGEC ≈ 927).
    assert n >= 800, f"объединённый банк слишком маленький: {n} пар"
    bank.build_index()
    # Должны находить пары на любую разумную русскую фразу.
    hits = bank.search("согласование числа в существительных", top_k=3)
    assert len(hits) == 3
    for _, pair in hits:
        assert pair.wrong and pair.right and pair.wrong != pair.right
