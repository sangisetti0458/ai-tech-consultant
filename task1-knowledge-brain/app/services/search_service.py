import json

import numpy as np

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Article, DocumentChunk
from app.services.embedding_service import embed_text


def search_knowledge(
    db: Session,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """Retrieve the most semantically similar indexed chunks."""

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    if top_k < 1 or top_k > 20:
        raise ValueError("top_k must be between 1 and 20.")

    query_embedding = np.asarray(
        embed_text(query),
        dtype=np.float32,
    )

    rows = db.execute(
        select(DocumentChunk, Article)
        .join(
            Article,
            DocumentChunk.article_id == Article.id,
        )
        .where(Article.status == "active")
    ).all()

    scored_results = []

    for chunk, article in rows:
        embedding = np.asarray(
            json.loads(chunk.embedding_json),
            dtype=np.float32,
        )

        if embedding.shape != query_embedding.shape:
            continue

        score = float(np.dot(query_embedding, embedding))

        scored_results.append({
            "article_id": article.id,
            "title": article.title,
            "url": article.url,
            "chunk_id": chunk.id,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "score": score,
        })

    scored_results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return scored_results[:top_k]