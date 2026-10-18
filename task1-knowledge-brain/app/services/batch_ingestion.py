import logging

from sqlalchemy.orm import Session

from app.ingestion.downloader import ArticleDownloader
from app.services.article_service import save_article

logger = logging.getLogger(__name__)


async def ingest_batch(
    db: Session,
    urls: list[str],
) -> dict:
    """
    Download a batch of articles and save successful results.

    Each article is processed independently so that one failure
    does not prevent the remaining articles from being saved.
    """

    downloader = ArticleDownloader()

    results = {
        "requested": len(urls),
        "successful": 0,
        "failed": 0,
        "new_articles": 0,
        "updated_articles": 0,
        "unchanged_articles": 0,
        "errors": [],
        "articles": [],
    }

    # Download and parse the requested URLs.
    # Failures are logged by the downloader.
    downloaded_articles = await downloader.download_many(urls)

    results["failed"] = len(urls) - len(downloaded_articles)

    for article_data in downloaded_articles:
        try:
            saved = save_article(db, article_data)

            results["successful"] += 1

            if saved["is_new"]:
                results["new_articles"] += 1
            elif saved["content_changed"]:
                results["updated_articles"] += 1
            else:
                results["unchanged_articles"] += 1

            results["articles"].append(saved)

        except Exception as exc:
            db.rollback()

            results["failed"] += 1

            results["errors"].append({
                "url": article_data["url"],
                "error": str(exc),
            })

            logger.exception(
                "Failed to save article: %s",
                article_data["url"],
            )

    return results