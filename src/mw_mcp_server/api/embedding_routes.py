"""
Embeddings Routes

This module exposes endpoints for:
- Querying embedding index statistics
- Updating page embeddings
- Deleting page embeddings

These endpoints are designed to be invoked by trusted MediaWiki services
for synchronizing wiki page content into the MCP vector search layer.

Uses PostgreSQL + pgvector for storage and similarity search.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated, Optional
from datetime import datetime, timezone


from .models import (
    EmbeddingUpdatePageRequest,
    EmbeddingDeletePageRequest,
    EmbeddingStatsResponse,
    OperationResult,
)
from .dependencies import get_vector_store
from ..auth.security import require_scopes
from ..auth.models import UserContext
from ..db import VectorStore
from ..embeddings.queue import EmbeddingJob, QueueFullError, embedding_queue
import logging

logger = logging.getLogger(__name__)

# Suggested client back-off when the embedding queue is full.
_QUEUE_FULL_RETRY_AFTER_SECONDS = 60


def _parse_last_modified(raw: Optional[str]) -> Optional[datetime]:
    """Parse a MediaWiki (YYYYMMDDHHMMSS) or ISO-8601 timestamp to naive UTC.

    Storage columns are ``timestamp without time zone`` holding UTC, so
    offset-aware inputs are normalised rather than passed through (asyncpg
    rejects aware datetimes for naive columns). Unparseable input -> None.
    """
    if not raw:
        return None
    ts = raw.strip()
    try:
        if len(ts) == 14 and ts.isdigit():
            return datetime.strptime(ts, "%Y%m%d%H%M%S")
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


router = APIRouter(prefix="/embeddings", tags=["embeddings"])


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@router.get(
    "/stats",
    response_model=EmbeddingStatsResponse,
    summary="Get embedding index statistics",
)
async def get_embedding_stats(
    user: Annotated[UserContext, Depends(require_scopes("embeddings"))],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
) -> EmbeddingStatsResponse:
    """
    Return current embedding index statistics for the user's wiki.
    """
    stats = await vector_store.get_stats(user.wiki_id)
    return EmbeddingStatsResponse(**stats)


@router.post(
    "/page",
    summary="Create or update a page embedding",
    response_model=OperationResult,
)
async def update_page_embedding(
    req: EmbeddingUpdatePageRequest,
    user: Annotated[UserContext, Depends(require_scopes("embeddings"))],
) -> OperationResult:
    """
    Enqueue a job to create or update embeddings for a wiki page.
    Returns immediately with status='queued'.

    If the page already has a pending job, its payload is replaced (latest
    content wins). Returns 503 with Retry-After when the queue is full so the
    caller retries later instead of the update being silently dropped.
    """
    job = EmbeddingJob(
        wiki_id=user.wiki_id,
        title=req.title,
        content=req.content,
        namespace=req.namespace,
        last_modified=_parse_last_modified(req.last_modified),
        rev_id=req.rev_id,
    )

    try:
        qsize = await embedding_queue.enqueue(job)
    except QueueFullError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding queue is full; retry later.",
            headers={"Retry-After": str(_QUEUE_FULL_RETRY_AFTER_SECONDS)},
        )
    except Exception:
        logger.exception("Failed to enqueue embedding job")
        raise

    return OperationResult(
        status="queued",
        count=0,
        details={"queue_size": qsize}
    )


@router.delete(
    "/page",
    summary="Delete embeddings for a wiki page",
    response_model=OperationResult,
)
async def delete_page_embedding(
    req: EmbeddingDeletePageRequest,
    user: Annotated[UserContext, Depends(require_scopes("embeddings"))],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
) -> OperationResult:
    """
    Delete all embeddings for a given wiki page.

    Also cancels any pending update for the page in the same transaction, so
    an update queued before this delete can't re-insert the page afterwards.
    """
    count = await vector_store.delete_page_and_pending_job(user.wiki_id, req.title)
    await vector_store.commit()

    return OperationResult(
        status="deleted",
        count=count
    )
