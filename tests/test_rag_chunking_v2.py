from shared.rag_chunking import safe_chunk_text


def test_one_huge_rtf_paragraph_never_exceeds_limit():
    text = ("Приказ устанавливает порядок ведения учета и оформления документов. " * 500).strip()
    chunks = list(safe_chunk_text(text, chunk_chars=600, overlap=80))
    assert len(chunks) > 10
    assert all(0 < len(chunk) <= 600 for chunk in chunks)
    assert "Приказ устанавливает" in chunks[0]


def test_short_text_stays_single_chunk():
    assert list(safe_chunk_text("Короткий пункт.", chunk_chars=600)) == ["Короткий пункт."]
