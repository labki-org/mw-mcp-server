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

from fastapi import APIRouter, Depends
from typing import Annotated
from datetime import datetime


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
import logging

logger = logging.getLogger(__name__)

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
    Returns immediately with stats='queued'.
    """
    try:
        # Parse timestamp first (synchronous check)
        last_modified = None
        if req.last_modified:
            ts = req.last_modified.strip()
            try:
                if len(ts) == 14 and ts.isdigit():
                    last_modified = datetime.strptime(ts, "%Y%m%d%H%M%S")
                else:
                    last_modified = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        # Create job
        from ..embeddings.queue import EmbeddingJob, embedding_queue
        
        job = EmbeddingJob(
            wiki_id=user.wiki_id,
            title=req.title,
            content=req.content,
            namespace=req.namespace,
            last_modified=last_modified
        )

        # Enqueue
        qsize = await embedding_queue.enqueue(job)

        # Return immediate success
        return OperationResult(
            status="queued",
            count=0,
            details={"queue_size": qsize}
        )
    except Exception:
        logger.exception("Failed to enqueue embedding job")
        raise


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
    """
    count = await vector_store.delete_page(user.wiki_id, req.title)
    await vector_store.commit()

    return OperationResult(
        status="deleted",
        count=count
    )
