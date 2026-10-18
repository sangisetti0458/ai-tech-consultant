def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[str]:
    """
    Split text into overlapping chunks.

    chunk_size and overlap are measured in characters,
    not tokens.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    normalized_text = " ".join(text.split())

    if not normalized_text:
        return []

    chunks = []
    start = 0

    while start < len(normalized_text):
        end = min(start + chunk_size, len(normalized_text))

        if end < len(normalized_text):
            # Prefer breaking at a word boundary.
            boundary = normalized_text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = normalized_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(normalized_text):
            break

        start = end - overlap

    return chunks