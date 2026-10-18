import numpy as np
import pytest

from app.services.embedding_service import (
    cosine_similarity,
    embed_text,
    embed_texts,
)


def test_embed_text_returns_normalized_vector():
    embedding = embed_text(
        "Configure payment settings on your website."
    )

    assert len(embedding) == 384
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)


def test_embed_text_rejects_empty_input():
    with pytest.raises(ValueError):
        embed_text("   ")


def test_embed_texts_returns_one_vector_per_input():
    embeddings = embed_texts([
        "Configure payment settings.",
        "Manage website membership plans.",
    ])

    assert len(embeddings) == 2
    assert all(len(embedding) == 384 for embedding in embeddings)


def test_cosine_similarity_of_identical_vectors():
    vector = [1.0, 0.0, 0.0]

    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_cosine_similarity_rejects_mismatched_dimensions():
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])