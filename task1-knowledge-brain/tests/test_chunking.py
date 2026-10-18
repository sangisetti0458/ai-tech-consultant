import pytest

from app.services.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []


def test_short_text_returns_one_chunk():
    result = chunk_text("Hello world")

    assert result == ["Hello world"]


def test_long_text_creates_multiple_chunks():
    text = "word " * 500

    result = chunk_text(
        text,
        chunk_size=200,
        overlap=30,
    )

    assert len(result) > 1
    assert all(result)


def test_invalid_chunk_size_raises_error():
    with pytest.raises(ValueError):
        chunk_text("Some text", chunk_size=0)


def test_invalid_overlap_raises_error():
    with pytest.raises(ValueError):
        chunk_text("Some text", chunk_size=100, overlap=100)