from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Reduce unnecessary CPU thread usage
torch.set_num_threads(1)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""
    return SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )


def embed_text(text: str) -> list[float]:
    """Generate an embedding for a single text."""
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    model = get_embedding_model()

    embedding = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return embedding.astype(np.float32).tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple texts."""
    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError("Texts cannot contain empty strings.")

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=8,
        show_progress_bar=False,
    )

    return embeddings.astype(np.float32).tolist()


def cosine_similarity(
    embedding_a: list[float],
    embedding_b: list[float],
) -> float:
    """Calculate cosine similarity between two vectors."""
    vector_a = np.asarray(embedding_a, dtype=np.float32)
    vector_b = np.asarray(embedding_b, dtype=np.float32)

    if vector_a.ndim != 1 or vector_b.ndim != 1:
        raise ValueError("Embeddings must be one-dimensional.")

    if vector_a.shape != vector_b.shape:
        raise ValueError("Embedding dimensions must match.")

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Embeddings must not be zero vectors.")

    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))