import pytest

from app.ingestion.article_parser import extract_article


def test_extract_article_content():
    html = """
    <html>
        <head>
            <title>Payment Settings</title>
        </head>
        <body>
            <nav>Navigation content</nav>
            <main>
                <h1>Payment Settings</h1>
                <h2>Configuration</h2>
                <p>Open your payment settings.</p>
                <img src="/images/payment.png" alt="Payment settings">
            </main>
        </body>
    </html>
    """

    result = extract_article(
        "https://support.brilliantdirectories.com/support/solutions/test",
        html,
    )

    assert result["title"] == "Payment Settings"
    assert "Open your payment settings." in result["text"]
    assert len(result["images"]) == 1
    assert result["images"][0]["alt"] == "Payment settings"


def test_reject_external_host():
    with pytest.raises(ValueError):
        extract_article(
            "https://example.com/article",
            "<html><body>Test</body></html>",
        )


def test_content_hash_is_generated():
    result = extract_article(
        "https://support.brilliantdirectories.com/support/solutions/test",
        "<html><body><main><p>Example text</p></main></body></html>",
    )

    assert len(result["content_hash"]) == 64