from sqlalchemy.orm import Session

from app.services.search_service import search_knowledge


def answer_question(
    db: Session,
    question: str,
    top_k: int = 5,
) -> dict:
    """
    Generate a simple extractive answer using retrieved documentation.
    No paid LLM API is required.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    sources = search_knowledge(
        db=db,
        query=question,
        top_k=top_k,
    )

    if not sources:
        return {
            "status": "insufficient_evidence",
            "question": question,
            "answer": (
                "I could not find relevant documentation "
                "to answer this question reliably."
            ),
            "sources": [],
        }

    # Build an extractive response from retrieved documentation.
    answer_parts = []

    for index, source in enumerate(sources, start=1):
        text = source.get("text", "").strip()

        if text:
            answer_parts.append(
                f"[{index}] {source['title']}: {text}"
            )

    if not answer_parts:
        return {
            "status": "insufficient_evidence",
            "question": question,
            "answer": (
                "Relevant documents were retrieved, but "
                "they did not contain usable text."
            ),
            "sources": [],
        }

    return {
        "status": "answered",
        "question": question,
        "answer": "\n\n".join(answer_parts),
        "sources": [
            {
                "source_id": index,
                "article_id": source["article_id"],
                "title": source["title"],
                "url": source["url"],
                "score": source["score"],
            }
            for index, source in enumerate(sources, start=1)
        ],
        "model": "extractive",
    }