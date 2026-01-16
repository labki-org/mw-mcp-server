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
from typing import List, Annotated
from datetime import datetime


from .models import (
    EmbeddingUpdatePageRequest,
    EmbeddingDeletePageRequest,
    EmbeddingStatsResponse,
    OperationResult,
)
from .dependencies import get_vector_store, get_embedder
from ..auth.security import require_scopes
from ..auth.models import UserContext
from ..db import VectorStore
from ..embeddings.embedder import Embedder

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def _chunk_text(text: str, min_length: int = 10) -> List[str]:
    """
    Return the full text as a single chunk if it meets minimum length requirements.
    
    Parameters
    ----------
    text : str
        Raw page text.
    min_length : int
        Minimum character length required for a chunk to be embedded.

    Returns
    -------
    List[str]
        List containing the single trimmed text, or empty list if too short.
    """
    trimmed = text.strip()
    if len(trimmed) < min_length:
        return []

    # Simple constraint: OpenAI text-embedding-3-large limit is 8191 tokens.
    # ~4 chars per token -> ~32k chars. We use 25k chars as a safe upper bound.
    MAX_CHUNK_SIZE = 25000
    
    if len(trimmed) <= MAX_CHUNK_SIZE:
        return [trimmed]

    chunks = []
    # Naive chunking by length
    for i in range(0, len(trimmed), MAX_CHUNK_SIZE):
        chunks.append(trimmed[i : i + MAX_CHUNK_SIZE])
    
    return chunks


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
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> OperationResult:
    """
    Create or update embeddings for a wiki page.

    Workflow
    --------
    1. Chunk page content into paragraphs.
    2. Delete any existing embeddings for the page.
    3. Embed the new content.
    4. Add new vectors to the database.
    """

    # -------------------------------------------------------------
    # 1. Chunk Content
    # -------------------------------------------------------------
    text_chunks = _chunk_text(req.content)

    # -------------------------------------------------------------
    # 2. Delete Existing Page Embeddings
    # -------------------------------------------------------------
    await vector_store.delete_page(user.wiki_id, req.title)

    # If no valid content remains after chunking, exit early
    if not text_chunks:
        return OperationResult(
            status="deleted",
            count=0,
            details={"reason": "empty_content_after_chunking"}
        )

    # -------------------------------------------------------------
    # 3. Embed & Add to Index
    # -------------------------------------------------------------
    embeddings = await embedder.embed(text_chunks)
    
    # Parse last_modified timestamp
    last_modified = None
    if req.last_modified:
        try:
            last_modified = datetime.fromisoformat(req.last_modified.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass

    count = await vector_store.add_documents(
        wiki_id=user.wiki_id,
        page_titles=[req.title] * len(text_chunks),
        section_ids=[None] * len(text_chunks),
        namespaces=[req.namespace] * len(text_chunks),
        embeddings=embeddings,
        last_modified=last_modified,
    )

    return OperationResult(
        status="updated",
        count=count
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
    """
    count = await vector_store.delete_page(user.wiki_id, req.title)

    return OperationResult(
        status="deleted",
        count=count
    )
