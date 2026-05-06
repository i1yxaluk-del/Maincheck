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


def test_lexify_admin_bank_file_is_valid() -> None:
    """v1.7.2: банк морфо-агрементов из alexanderpl/Lexify_RuGEC парсится."""
    here = Path(__file__).resolve().parent
    lexify = here.parent / "server" / "shared" / "gec_seed" / "lexify_admin.jsonl"
    assert lexify.exists(), f"lexify_admin банк отсутствует: {lexify}"
    bank = GecBank(HashingEmbedder(dim=64))
    n = bank.load_jsonl(lexify)
    # Целевой размер фильтра — 1500-2000 пар (4 категории × 1500 каждая
    # с балансом по реально найденным).
    assert n >= 1000, f"lexify_admin банк слишком маленький: {n} пар"
    bank.build_index()
    # В банке должны быть представлены ВСЕ 4 морфо-категории.
    rules = {p.rule for p in (e.pair for e in bank._entries)}
    assert "Lexify-case_agreement" in rules
    assert "Lexify-number_agreement" in rules
    assert "Lexify-verb_form" in rules
    assert "Lexify-case_and_number" in rules


def test_full_bank_v1_7_2_helps_kvartalakh_query() -> None:
    """v1.7.2 регрессия-тест: full-stack банк (3 файла) находит relevant
    пары для prod-кейса «Во 2-м кварталах планируется направление...»

    До v1.7.2 (только seed + extended, 927 пар) топ-3 для этого запроса
    были: «меч и лук и стрелы», «не соприкасающиеся темы», «3 1/2-тысячный
    коллектив» — все нерелевантные.

    После v1.7.2 топ-3 должны содержать минимум 1 пару из Lexify-*
    (морфо-агременты) — это и есть основной выигрыш ретривала.
    """
    here = Path(__file__).resolve().parent
    seed = here.parent / "server" / "shared" / "gec_seed" / "gec_bank.jsonl"
    extended = here.parent / "server" / "shared" / "gec_seed" / "gec_bank_extended.jsonl"
    lexify = here.parent / "server" / "shared" / "gec_seed" / "lexify_admin.jsonl"
    bank = GecBank(HashingEmbedder(dim=64), bm25_tokenizer="both")
    n = bank.load_jsonl(seed, extended, lexify)
    assert n >= 2000, f"full-stack банк слишком маленький: {n} пар"
    bank.build_index()

    query = "Во 2-м кварталах планируется направление материалов проверки"
    results = bank.search_sparse(query, top_k=5)
    rules = [pair.rule for _, pair in results]
    # Минимум 1 Lexify-* пара в топ-5 — иначе ретривал не починен.
    lexify_hits = [r for r in rules if r.startswith("Lexify-")]
    assert lexify_hits, (
        f"Lexify-пары не попали в топ-5 для prod-запроса. "
        f"Топ-5 rules: {rules}. Расширение банка не работает."
    )


# ─── v1.6.7: BM25 + hybrid retrieval ──────────────────────────────


def test_tokenize_ru_basic() -> None:
    """Токенизатор BM25: lower, ё→е, отбрасывает пунктуацию."""
    from shared.gec_bank import _tokenize_ru

    tokens = _tokenize_ru("Стоимостей выполненной работ, путём!")
    assert tokens == ["стоимостей", "выполненной", "работ", "путем"]
    # «Запятая, точка. — Тире» — пунктуация и тире не выживают
    assert _tokenize_ru("«Запятая, точка. — Тире»") == ["запятая", "точка", "тире"]
    # ё → е
    assert _tokenize_ru("Проведённый") == _tokenize_ru("Проведенный")


def test_bm25_index_score_known_doc() -> None:
    """BM25 должен поднимать документ с запрошенными редкими токенами."""
    from shared.gec_bank import _BM25Index, _tokenize_ru

    docs = [
        _tokenize_ru("Стоимостей выполненной работ путём применения завышенных расценок"),
        _tokenize_ru("Совершенно посторонний документ про административные процедуры"),
        _tokenize_ru("Согласно приказа был утверждён план работы отдела"),
    ]
    idx = _BM25Index(docs)
    # Запрос с редкими словоформами из dок 0 → он должен быть top-1
    scores = idx.score(_tokenize_ru("Стоимостей выполненной работ"))
    assert max(range(len(scores)), key=lambda i: scores[i]) == 0
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_bm25_index_unknown_term_does_not_crash() -> None:
    """Запрос с термином, которого нет в словаре, не падает и не даёт скор."""
    from shared.gec_bank import _BM25Index, _tokenize_ru

    idx = _BM25Index([_tokenize_ru("привет мир")])
    scores = idx.score(_tokenize_ru("незнакомое слово xyzabc"))
    assert scores == [0.0]


def test_bm25_index_empty_collection() -> None:
    """Пустая коллекция: avgdl=0, score не падает, возвращает пусто."""
    from shared.gec_bank import _BM25Index

    idx = _BM25Index([])
    assert idx.score(["foo"]) == []


# ─── v1.7: char-trigram BM25 ────────────────────────────────────────


def test_tokenize_ru_trigram_basic() -> None:
    """Char-trigram токенизатор: padding `$`, lower, ё→е, размер 3."""
    from shared.gec_bank import _tokenize_ru_trigram

    # «ок» (2 символа) → padded «$ок$» (4 символа), 4-3+1=2 триграммы
    assert _tokenize_ru_trigram("ок") == ["$ок", "ок$"]

    # «стои» (4 символа) → padded «$стои$», 6-3+1=4 триграммы
    assert _tokenize_ru_trigram("стои") == ["$ст", "сто", "тои", "ои$"]

    # ё нормализуется к е
    assert _tokenize_ru_trigram("прошёл") == _tokenize_ru_trigram("прошел")

    # Регистр не важен
    assert _tokenize_ru_trigram("СТО") == _tokenize_ru_trigram("сто")

    # Пунктуация и пробелы — разделители слов, не попадают в триграммы
    out = _tokenize_ru_trigram("Привет, мир!")
    assert "$пр" in out
    assert "ет$" in out  # «привет» закрывается на $
    assert "$ми" in out
    # Не должно быть триграмм с запятой
    assert all("," not in t for t in out)
    assert all(" " not in t for t in out)


def test_tokenize_ru_trigram_morphological_overlap() -> None:
    """Главная польза trigram-токенизатора: разные словоформы одного
    корня делят триграммы. word-токенизатор их различает, trigram —
    объединяет."""
    from shared.gec_bank import _tokenize_ru_trigram

    a = set(_tokenize_ru_trigram("стоимостей"))
    b = set(_tokenize_ru_trigram("стоимости"))
    # Word-tokenizer вернул бы 0 совпадений; у trigram-токенизатора
    # должно быть много общих триграмм (общий корень «стоимост-»).
    overlap = a & b
    assert len(overlap) >= 6, f"ожидалось ≥6 общих триграмм, получено {overlap}"


def test_bm25_trigram_index_finds_morphologically_close() -> None:
    """BM25 на trigram-токенах должен находить doc с похожей
    словоформой (разные падежи/числа одного корня)."""
    from shared.gec_bank import _BM25Index, _tokenize_ru_trigram

    docs = [
        _tokenize_ru_trigram("выполненных работ путём применения"),
        _tokenize_ru_trigram("совершенно посторонний документ"),
        _tokenize_ru_trigram("согласно приказа утверждён план"),
    ]
    idx = _BM25Index(docs)
    # Запрос «выполненной работы» — другая падежно-числовая форма
    # того же корня. Word-BM25 не нашёл бы — у «выполненной» и
    # «выполненных» нет общих токенов. Trigram должен найти.
    scores = idx.score(_tokenize_ru_trigram("выполненной работы"))
    top = max(range(len(scores)), key=lambda i: scores[i])
    assert top == 0, f"trigram BM25 должен поднять doc 0, поднял {top}: {scores}"
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_gec_bank_tokenizer_mode_default_is_word(sample_bank_file: Path) -> None:
    """Без явного аргумента GecBank создаётся в режиме `word` —
    backward-compatible с v1.6: только word-индекс, нет trigram."""
    bank = GecBank(HashingEmbedder(dim=64))
    bank.load_jsonl(sample_bank_file)
    bank.build_index()
    assert bank.bm25_tokenizer == "word"
    assert bank._bm25 is not None
    assert bank._bm25_trigram is None


def test_gec_bank_tokenizer_mode_trigram(sample_bank_file: Path) -> None:
    """В режиме `trigram` строится только trigram-индекс."""
    bank = GecBank(HashingEmbedder(dim=64), bm25_tokenizer="trigram")
    bank.load_jsonl(sample_bank_file)
    bank.build_index()
    assert bank.bm25_tokenizer == "trigram"
    assert bank._bm25 is None
    assert bank._bm25_trigram is not None


def test_gec_bank_tokenizer_mode_both(sample_bank_file: Path) -> None:
    """В режиме `both` строятся оба индекса параллельно."""
    bank = GecBank(HashingEmbedder(dim=64), bm25_tokenizer="both")
    bank.load_jsonl(sample_bank_file)
    bank.build_index()
    assert bank.bm25_tokenizer == "both"
    assert bank._bm25 is not None
    assert bank._bm25_trigram is not None


def test_gec_bank_tokenizer_mode_invalid_falls_back_to_word(sample_bank_file: Path) -> None:
    """Некорректный mode — мягкий даунгрейд до word, без падений."""
    bank = GecBank(HashingEmbedder(dim=64), bm25_tokenizer="hieroglyph")
    bank.load_jsonl(sample_bank_file)
    bank.build_index()
    assert bank.bm25_tokenizer == "word"
    assert bank._bm25 is not None
    assert bank._bm25_trigram is None


def test_search_sparse_with_trigram_finds_morphological_match(sample_bank_file: Path) -> None:
    """Главный кейс v1.7: запрос с другой словоформой того же корня.
    word-only sparse не находит, trigram/both — находят.

    Берём пару из банка с «приказа» (gent). Word-BM25 для запроса с
    «приказе» (locv) должен дать score=0 (нет общих словоформ).
    Trigram-BM25 — должен поднять эту пару (общие триграммы корня
    «приказ-»).
    """
    word_bank = GecBank(HashingEmbedder(dim=64), bm25_tokenizer="word")
    word_bank.load_jsonl(sample_bank_file)
    word_bank.build_index()
    # «приказе» нет в банке как словоформа, но есть «приказа», «приказу».
    word_hits = word_bank.search_sparse("Решение о приказе вынесено", top_k=3)
    word_rules = {p.rule for _, p in word_hits}

    tri_bank = GecBank(HashingEmbedder(dim=64), bm25_tokenizer="trigram")
    tri_bank.load_jsonl(sample_bank_file)
    tri_bank.build_index()
    tri_hits = tri_bank.search_sparse("Решение о приказе вынесено", top_k=3)
    tri_rules = {p.rule for _, p in tri_hits}

    # trigram должен покрыть хотя бы то же или больше — главное,
    # вытаскивает пару с «приказ-» корнем («Падежное управление»).
    assert "Падежное управление" in tri_rules, (
        f"trigram не нашёл «приказ-» матч. word={word_rules}, tri={tri_rules}"
    )


def test_search_sparse_both_sums_scores(sample_bank_file: Path) -> None:
    """В режиме `both` скоры word и trigram суммируются. Проверяем
    что результат ранжирования согласован — top-1 пара должна
    получить высокий score (>= max from word-only)."""
    word_bank = GecBank(HashingEmbedder(dim=64), bm25_tokenizer="word")
    word_bank.load_jsonl(sample_bank_file)
    word_bank.build_index()
    both_bank = GecBank(HashingEmbedder(dim=64), bm25_tokenizer="both")
    both_bank.load_jsonl(sample_bank_file)
    both_bank.build_index()

    q = "Согласно приказа был утверждён план."
    word_hits = word_bank.search_sparse(q, top_k=1)
    both_hits = both_bank.search_sparse(q, top_k=1)
    # На точном запросе и word, и both должны вернуть ту же пару топом.
    assert word_hits and both_hits
    assert word_hits[0][1].rule == both_hits[0][1].rule
    # both score >= word score (т.к. тригграммы добавляют положительный
    # вклад на любых пересекающихся корнях).
    assert both_hits[0][0] >= word_hits[0][0]


def test_gec_bank_stats_includes_trigram_terms(sample_bank_file: Path) -> None:
    """stats() в режиме `both` показывает количество триграмм."""
    bank = GecBank(HashingEmbedder(dim=64), bm25_tokenizer="both")
    bank.load_jsonl(sample_bank_file)
    bank.build_index()
    s = bank.stats()
    assert s["bm25_tokenizer"] == "both"
    assert s["bm25_terms"] > 0
    assert s["bm25_trigram_terms"] > 0
    # У trigram-индекса всегда больше «терминов» (триграмм >> словоформ).
    assert s["bm25_trigram_terms"] > s["bm25_terms"]


def test_search_sparse_returns_lexically_close(sample_bank_file: Path) -> None:
    """search_sparse находит пару по точному совпадению словоформ,
    минуя dense (где hashing-embedder может не сойтись)."""
    bank = _make_bank(sample_bank_file)
    # «согласно приказа» — пара из банка, ищем по тем же словоформам
    hits = bank.search_sparse("Согласно приказа был утверждён план", top_k=2)
    assert len(hits) >= 1
    rules = [p.rule for _, p in hits]
    assert "Падежное управление" in rules
    # Скор > 0
    assert all(s > 0.0 for s, _ in hits)


def test_search_sparse_empty_when_no_overlap(sample_bank_file: Path) -> None:
    """Если в запросе нет ни одного слова из банка — sparse возвращает пусто."""
    bank = _make_bank(sample_bank_file)
    hits = bank.search_sparse("zzz qqq xxx aaa", top_k=3)
    assert hits == []


def test_search_hybrid_combines_dense_and_sparse(sample_bank_file: Path) -> None:
    """Hybrid (RRF dense + sparse) выдаёт пару, релевантную хотя бы одному
    из сигналов."""
    bank = _make_bank(sample_bank_file)
    hits = bank.search_hybrid("Согласно распоряжения был выпущен приказ", top_k=2)
    assert len(hits) == 2
    # Все score > 0 (RRF никогда не даёт 0 для не-пустого пересечения)
    assert all(s > 0.0 for s, _ in hits)
    rules = [p.rule for _, p in hits]
    # Лексический сигнал «приказа» / «согласно» однозначно тащит падежное
    # управление в top-2.
    assert "Падежное управление" in rules


def test_search_hybrid_empty_bank() -> None:
    """Hybrid на пустом банке — пустой список (не падает)."""
    bank = GecBank(HashingEmbedder(dim=32))
    assert bank.search_hybrid("любой запрос", top_k=5) == []


def test_search_hybrid_falls_back_when_no_bm25(sample_bank_file: Path) -> None:
    """Если BM25-индекс не построен (например, build_bm25_index не вызывался),
    hybrid не падает и возвращает результаты по dense."""
    bank = GecBank(HashingEmbedder(dim=64))
    bank.load_jsonl(sample_bank_file)
    bank.build_index()
    # Принудительно сбрасываем BM25 — имитируем старый кэш без sparse.
    bank._bm25 = None
    hits = bank.search_hybrid("Согласно распоряжения", top_k=2)
    # Возвращает то же, что dense — без падений.
    assert len(hits) == 2
    assert all(s > 0.0 for s, _ in hits)


def test_search_hybrid_outperforms_dense_on_long_admin_text() -> None:
    """Регрессионный тест на проблему v1.6.6 → v1.6.7:

    На длинном административном тексте dense (hashing surrogate)
    «размывается» предметной лексикой и уводит ранжирование от
    реальной грамматической ошибки. BM25 ловит редкие словоформы
    напрямую. Hybrid должен быть как минимум не хуже dense (то есть
    релевантная пара должна быть в top-3).

    Здесь воспроизводим симптом на крошечной коллекции, чтобы тест
    был детерминированным и быстрым.
    """
    pairs = [
        # Пара, которая ДОЛЖНА быть найдена: содержит «выполненной»
        GecPair(
            wrong="Стоимостей выполненной работ",
            right="Стоимости выполненных работ",
            rule="Согласование числа существительного и причастия",
        ),
        # Шум: про административный домен, но без грамматических совпадений
        GecPair(
            wrong="Государственный контракт КС-2 на сумму более 300 тыс рублей",
            right="Государственный контракт КС-2 на сумму более 300 тыс. рублей",
            rule="Сокращение",
        ),
        GecPair(
            wrong="Подписан акт освидетельствования скрытых работ задним числом",
            right="Подписан акт освидетельствования скрытых работ «задним числом»",
            rule="Кавычки при выражении",
        ),
    ]
    bank = GecBank(HashingEmbedder(dim=64))
    bank._entries = [
        __import__("shared.gec_bank", fromlist=["_Indexed"])._Indexed(
            pair=p, vec=[], norm=1.0,
        )
        for p in pairs
    ]
    bank.build_index()
    query = (
        "Установлены факты необоснованного завышения стоимостей выполненной "
        "работ, путём применения завышенных расценок"
    )
    hits = bank.search_hybrid(query, top_k=3)
    rules = [p.rule for _, p in hits]
    # Целевая пара должна быть в top-1 (BM25 ловит «стоимостей» и «выполненной»).
    assert rules[0] == "Согласование числа существительного и причастия", (
        f"hybrid не вытащил релевантную пару в top-1: {rules}"
    )


def test_metrics_exposes_bm25_terms(sample_bank_file: Path) -> None:
    """stats() после build_index содержит число BM25-терминов > 0."""
    bank = _make_bank(sample_bank_file)
    s = bank.stats()
    assert s.get("bm25_terms", 0) > 0


def test_search_hybrid_deterministic_tie_break(sample_bank_file: Path) -> None:
    """v1.6.9: при равном RRF-скоре тай-брейк по индексу пары в банке.

    Регрессионный тест на наблюдение из v1.6.8 prod (5 мая 2026): тот же
    КС-2 текст с тем же hybrid-индексом возвращал РАЗНЫЕ top-1 пары между
    запусками сервера. Корень — `set(candidates)` итерируется в
    непредсказуемом порядке между процессами, и при ничейных RRF-скорах
    стабильная сортировка фиксировала случайный набор индексов.

    Фикс: в search_hybrid сортируем по `(-score, idx)` — при равном score
    меньший индекс выше.
    """
    bank = _make_bank(sample_bank_file)
    # Запрос, который вряд ли даёт чёткого победителя — много пар получают
    # схожие скоры по обоим сигналам.
    query = "тестовый запрос без явных лексических совпадений"
    runs = [bank.search_hybrid(query, top_k=5) for _ in range(5)]
    # Все запуски возвращают идентичную последовательность пар.
    rules_per_run = [tuple(p.rule for _, p in r) for r in runs]
    assert len(set(rules_per_run)) == 1, (
        f"Hybrid retrieval недетерминирован: {rules_per_run}"
    )
