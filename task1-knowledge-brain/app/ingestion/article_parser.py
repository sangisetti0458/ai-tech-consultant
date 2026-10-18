import hashlib

from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup


ALLOWED_HOST = "support.brilliantdirectories.com"


def extract_article(url: str, html: str) -> dict:
    """
    Extract normalized article content from an HTML page.

    This is an initial generic parser. We will refine its selectors
    after inspecting the actual help-center HTML.
    """

    parsed_url = urlparse(url)

    if (
        parsed_url.scheme not in {"http", "https"}
        or parsed_url.hostname != ALLOWED_HOST
        or parsed_url.username is not None
        or parsed_url.password is not None
    ):
        raise ValueError("URL is outside the approved help-center host.")

    soup = BeautifulSoup(html, "lxml")

    # Remove elements that generally do not contain article content.
    for element in soup.select(
        "script, style, nav, footer, header, noscript"
    ):
        element.decompose()

    # Extract the title.
    title_tag = soup.find("h1") or soup.find("title")

    title = (
        title_tag.get_text(" ", strip=True)
        if title_tag
        else ""
    )

    # Try common article-content containers.
    content = soup.select_one(
        "article, "
        ".article-body, "
        ".article-content, "
        ".solution-content, "
        ".content-body, "
        "main"
    )

    # Fall back to the remaining page body if no known container exists.
    if content is None:
        content = soup.body or soup

    # Extract headings and paragraphs in document order.
    sections = []

    for element in content.find_all(
        ["h1", "h2", "h3", "h4", "p", "li"],
        recursive=True,
    ):
        text = element.get_text(" ", strip=True)

        if not text:
            continue

        sections.append(
            {
                "type": element.name,
                "text": text,
            }
        )

    # Preserve image references and their available context.
    images = []

    for image in content.find_all("img"):
        src = image.get("src") or image.get("data-src")

        if not src:
            continue

        image_url = urljoin(url, src)

        if urlparse(image_url).scheme not in {"http", "https"}:
            continue

        images.append(
            {
                "url": image_url,
                "alt": image.get("alt", ""),
                "context": image.parent.get_text(" ", strip=True)
                if image.parent
                else "",
            }
        )

    # Build normalized text.
    text = "\n".join(section["text"] for section in sections)

    content_hash = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return {
        "url": url,
        "title": title,
        "text": text,
        "sections": sections,
        "images": images,
        "content_hash": content_hash,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }