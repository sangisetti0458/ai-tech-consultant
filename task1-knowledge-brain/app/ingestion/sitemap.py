import asyncio
import logging
import xml.etree.ElementTree as ET

from urllib.parse import urlparse

import httpx


logger = logging.getLogger(__name__)

SITEMAP_URL = (
    "https://support.brilliantdirectories.com/support/sitemap.xml"
)

ALLOWED_HOST = "support.brilliantdirectories.com"

REQUEST_TIMEOUT = 20.0
MAX_RETRIES = 3


def validate_article_url(url: str) -> bool:
    """Accept only HTTP(S) URLs from the approved help-center host."""

    try:
        parsed = urlparse(url)

        return (
            parsed.scheme in {"http", "https"}
            and parsed.hostname == ALLOWED_HOST
            and parsed.username is None
            and parsed.password is None
        )

    except ValueError:
        return False


def parse_sitemap(xml_content: str) -> list[str]:
    """Extract and deduplicate valid URLs from sitemap XML."""

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as exc:
        raise ValueError("The sitemap contains invalid XML.") from exc

    urls: set[str] = set()

    for element in root.iter():
        if element.tag.split("}")[-1] != "loc":
            continue

        if not element.text:
            continue

        url = element.text.strip()

        if validate_article_url(url):
            urls.add(url)

    return sorted(urls)


async def fetch_sitemap(
    sitemap_url: str = SITEMAP_URL,
) -> list[str]:
    """Fetch the sitemap with timeouts and bounded retries."""

    if not validate_article_url(sitemap_url):
        raise ValueError("The sitemap URL is not an approved URL.")

    timeout = httpx.Timeout(REQUEST_TIMEOUT)

    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=False,
        headers={
            "User-Agent": "AI-Tech-Consultant/0.1 (educational project)"
        },
    ) as client:

        for attempt in range(MAX_RETRIES):
            try:
                response = await client.get(sitemap_url)
                response.raise_for_status()

                urls = parse_sitemap(response.text)

                logger.info("Discovered %s URLs", len(urls))

                return urls

            except (httpx.HTTPError, ValueError):
                if attempt == MAX_RETRIES - 1:
                    logger.exception("Failed to fetch sitemap")
                    raise

                delay = 2 ** attempt
                await asyncio.sleep(delay)

    return []