import asyncio
import logging

import httpx

from app.ingestion.article_parser import extract_article
from app.ingestion.sitemap import validate_article_url


logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 20.0
MAX_RETRIES = 3
MAX_CONCURRENT_REQUESTS = 3

USER_AGENT = (
    "AI-Tech-Consultant/0.1 "
    "(educational project; contact: project-maintainer)"
)


class ArticleDownloader:
    def __init__(self):
        self.semaphore = asyncio.Semaphore(
            MAX_CONCURRENT_REQUESTS
        )

        self.timeout = httpx.Timeout(REQUEST_TIMEOUT)

    async def download_article(
        self,
        client: httpx.AsyncClient,
        url: str,
    ) -> dict:
        """Download and parse one approved help-center article."""

        if not validate_article_url(url):
            raise ValueError("URL is outside the approved host.")

        async with self.semaphore:
            for attempt in range(MAX_RETRIES):
                try:
                    response = await client.get(url)
                    response.raise_for_status()

                    content_type = response.headers.get(
                        "content-type", ""
                    ).lower()

                    if "text/html" not in content_type:
                        raise ValueError(
                            "The response is not an HTML document."
                        )

                    result = extract_article(
                        url=url,
                        html=response.text,
                    )

                    if not result["text"].strip():
                        raise ValueError(
                            "No readable article text was extracted."
                        )

                    logger.info(
                        "Extracted article: %s",
                        result["title"] or url,
                    )

                    return result

                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code

                    # Do not retry permanent client errors.
                    if 400 <= status < 500 and status != 429:
                        logger.warning(
                            "Article returned HTTP %s: %s",
                            status,
                            url,
                        )
                        raise

                    if attempt == MAX_RETRIES - 1:
                        logger.exception(
                            "Article download failed: %s",
                            url,
                        )
                        raise

                    delay = 2 ** attempt
                    await asyncio.sleep(delay)

                except httpx.TransportError:
                    if attempt == MAX_RETRIES - 1:
                        logger.exception(
                            "Network error downloading: %s",
                            url,
                        )
                        raise

                    await asyncio.sleep(2 ** attempt)

            raise RuntimeError("Article download failed.")

    async def download_many(
        self,
        urls: list[str],
    ) -> list[dict]:
        """Download a limited set of articles concurrently."""

        results = []

        headers = {
            "User-Agent": USER_AGENT,
        }

        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=False,
            headers=headers,
        ) as client:
            for url in urls:
                try:
                    result = await self.download_article(
                        client,
                        url,
                    )
                    results.append(result)

                except (httpx.HTTPError, ValueError) as exc:
                    logger.warning(
                        "Skipping article %s: %s",
                        url,
                        exc,
                    )

        return results