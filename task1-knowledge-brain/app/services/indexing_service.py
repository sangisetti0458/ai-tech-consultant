import json

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models import Article, DocumentChunk
from app.services.chunking import chunk_text
from app.services.embedding_service import embed_texts


def index_article(
    db: Session,
    article_id: int,
) -> dict:
    """Create or replace the searchable chunks for one article."""

    article = db.get(Article, article_id)

    if article is None:
        raise ValueError(f"Article {article_id} was not found.")

    chunks = chunk_text(article.text)

    embeddings = embed_texts(chunks)

    try:
        # Replace the existing index for this article.
        db.execute(
            delete(DocumentChunk).where(
                DocumentChunk.article_id == article_id
            )
        )

        for index, (text, embedding) in enumerate(
            zip(chunks, embeddings)
        ):
            db.add(
                DocumentChunk(
                    article_id=article_id,
                    chunk_index=index,
                    text=text,
                    embedding_json=json.dumps(embedding),
                )
            )

        db.commit()

    except Exception:
        db.rollback()
        raise

    return {
        "article_id": article_id,
        "title": article.title,
        "chunks_created": len(chunks),
    }


def index_all_articles(db: Session) -> dict:
    """Index every active article stored in the database."""

    article_ids = db.scalars(
        select(Article.id).where(Article.status == "active")
    ).all()

    results = []
    failures = []

    for article_id in article_ids:
        try:
            results.append(index_article(db, article_id))
        except Exception as exc:
            failures.append({
                "article_id": article_id,
                "error": str(exc),
            })

    return {
        "articles_found": len(article_ids),
        "articles_indexed": len(results),
        "failed": len(failures),
        "failures": failures,
        "results": results,
    }