from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Article


def save_article(db: Session, article_data: dict) -> dict:
    """
    Save an article or update its existing database record.

    Articles are uniquely identified by URL.
    The content hash helps identify changed content.
    """

    url = article_data["url"]

    article = db.scalar(
        select(Article).where(Article.url == url)
    )

    is_new = article is None
    content_changed = False

    if is_new:
        article = Article(
            url=url,
            title=article_data.get("title", ""),
            text=article_data.get("text", ""),
            content_hash=article_data["content_hash"],
            status="active",
        )

        db.add(article)
        content_changed = True

    else:
        content_changed = (
            article.content_hash != article_data["content_hash"]
        )

        article.title = article_data.get("title", "")
        article.text = article_data.get("text", "")
        article.content_hash = article_data["content_hash"]
        article.status = "active"

    db.commit()
    db.refresh(article)

    return {
        "id": article.id,
        "url": article.url,
        "title": article.title,
        "is_new": is_new,
        "content_changed": content_changed,
    }


def get_article(db: Session, article_id: int) -> Article | None:
    """Retrieve an article by its database ID."""

    return db.get(Article, article_id)


def list_articles(
    db: Session,
    limit: int = 20,
    offset: int = 0,
) -> list[Article]:
    """List active articles with pagination."""

    statement = (
        select(Article)
        .where(Article.status == "active")
        .order_by(Article.id)
        .offset(offset)
        .limit(limit)
    )

    return list(db.scalars(statement).all())