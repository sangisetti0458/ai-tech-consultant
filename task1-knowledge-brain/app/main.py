from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timezone
from app.ingestion.sitemap import fetch_sitemap
from app.ingestion.downloader import ArticleDownloader
from contextlib import asynccontextmanager
from app.services.answer_service import answer_question

from app.database import init_db

from fastapi import Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.article_service import (
    get_article,
    list_articles,
    save_article,
)

from app.services.batch_ingestion import ingest_batch

from app.services.indexing_service import (
    index_all_articles,
    index_article,
)

from app.services.search_service import search_knowledge

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(
    title="AI Tech Consultant - Knowledge Brain",
    description="API for product knowledge retrieval and AI-powered answers.",
    version="0.1.0",
    lifespan=lifespan,
)


class HealthResponse(BaseModel):
    status: str
    service: str


@app.get("/", tags=["General"])
async def root():
    return {
        "message": "AI Tech Consultant Knowledge Brain API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    return HealthResponse(
        status="healthy",
        service="knowledge-brain",
    )


@app.get("/api/v1/info", tags=["General"])
async def get_info():
    return {
        "service": "Knowledge Brain",
        "description": "Shared knowledge service for the AI Tech Consultant.",
        "features": [
            "Knowledge ingestion",
            "Document retrieval",
            "Grounded question answering",
            "Source citations",
            "Human escalation",
        ],
        "current_stage": "Initial API setup",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/v1/ingestion/sitemap", tags=["Ingestion"])
async def discover_sitemap_urls():
    """Discover article URLs from the approved help-center sitemap."""

    urls = await fetch_sitemap()

    return {
        "status": "completed",
        "source": "Brilliant Directories Solution Center",
        "total_urls": len(urls),
        "urls": urls,
    }

@app.post(
    "/api/v1/ingestion/test-articles",
    tags=["Ingestion"],
)
async def test_article_download():
    """Download a small, predefined article batch for validation."""

    urls = [
        "https://support.brilliantdirectories.com/support/solutions/12000001358",
    ]

    downloader = ArticleDownloader()
    articles = await downloader.download_many(urls)

    return {
        "status": "completed",
        "requested": len(urls),
        "successful": len(articles),
        "articles": articles,
    }


@app.post("/api/v1/articles/ingest", tags=["Articles"])
async def ingest_article(
    db: Session = Depends(get_db),
):
    """Download one test article and save it to the database."""

    urls = [
        "https://support.brilliantdirectories.com/support/solutions/12000001358",
    ]

    downloader = ArticleDownloader()
    articles = await downloader.download_many(urls)

    if not articles:
        raise HTTPException(
            status_code=502,
            detail="Unable to download a valid article.",
        )

    result = save_article(db, articles[0])

    return {
        "status": "saved",
        "article": result,
    }


@app.get("/api/v1/articles", tags=["Articles"])
def get_articles(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """List stored articles."""

    articles = list_articles(
        db=db,
        limit=limit,
        offset=offset,
    )

    return {
        "count": len(articles),
        "articles": [
            {
                "id": article.id,
                "url": article.url,
                "title": article.title,
                "content_hash": article.content_hash,
            }
            for article in articles
        ],
    }

@app.get("/api/v1/articles/{article_id}", tags=["Articles"])
def get_article_by_id(
    article_id: int,
    db: Session = Depends(get_db),
):
    """Retrieve the stored content of one article."""

    article = get_article(db, article_id)

    if article is None:
        raise HTTPException(
            status_code=404,
            detail="Article not found.",
        )

    return {
        "id": article.id,
        "url": article.url,
        "title": article.title,
        "text": article.text,
        "content_hash": article.content_hash,
        "status": article.status,
    }

@app.post(
    "/api/v1/ingestion/batch",
    tags=["Ingestion"],
)
async def run_batch_ingestion(
    batch_size: int = Query(default=5, ge=1, le=10),
    db: Session = Depends(get_db),
):
    """
    Ingest a small batch of URLs discovered from the sitemap.
    """

    urls = await fetch_sitemap()

    if not urls:
        raise HTTPException(
            status_code=502,
            detail="No valid URLs were discovered.",
        )

    selected_urls = urls[:batch_size]

    results = await ingest_batch(
        db=db,
        urls=selected_urls,
    )

    return {
        "status": "completed",
        "source": "Brilliant Directories Solution Center",
        **results,
    }


@app.post("/api/v1/index/rebuild", tags=["Search"])
def rebuild_search_index(
    db: Session = Depends(get_db),
):
    """Build the semantic search index from stored articles."""

    return index_all_articles(db)


@app.post("/api/v1/index/{article_id}", tags=["Search"])
def index_single_article(
    article_id: int,
    db: Session = Depends(get_db),
):
    """Index or refresh the searchable chunks for one article."""

    try:
        return index_article(db, article_id)

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


@app.get("/api/v1/search", tags=["Search"])
def semantic_search(
    q: str = Query(..., min_length=1, max_length=1000),
    top_k: int = Query(default=5, ge=1, le=20),
    db: Session = Depends(get_db),
):
    """Search indexed documentation using semantic similarity."""

    try:
        results = search_knowledge(
            db=db,
            query=q,
            top_k=top_k,
        )

        return {
            "query": q,
            "results_count": len(results),
            "results": results,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.get("/api/v1/ask", tags=["Knowledge Assistant"])
def ask_question(
    q: str = Query(
        ...,
        min_length=1,
        max_length=1000,
    ),
    top_k: int = Query(
        default=5,
        ge=1,
        le=10,
    ),
    db: Session = Depends(get_db),
):
    """Answer a question using retrieved product documentation."""

    try:
        return answer_question(
            db=db,
            question=q,
            top_k=top_k,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
        