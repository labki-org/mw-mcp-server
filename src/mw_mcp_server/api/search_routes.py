"""
Search Routes

This module defines vector-based semantic search endpoints backed by the
PostgreSQL + pgvector embedding index. These routes are typically invoked 
by the LLM during tool execution as well as by the MediaWiki client for 
direct user queries.
"""

from fastapi import APIRouter, Depends, status
from typing import List, Annotated

from .models import SearchRequest, SearchResult
from ..auth.security import require_scopes
from ..auth.models import UserContext
from ..db import VectorStore
from ..embeddings.embedder import Embedder
from .dependencies import get_vector_store, get_embedder

router = APIRouter(prefix="/search", tags=["search"])


@router.post(
    "/",
    response_model=List[SearchResult],
    summary="Vector-based semantic search",
    status_code=status.HTTP_200_OK,
)
async def search(
    req: SearchRequest,
    user: Annotated[UserContext, Depends(require_scopes("search"))],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> List[SearchResult]:
    """
    Perform a vector-based semantic search over embedded wiki content.

    Parameters
    ----------
    req : SearchRequest
        Contains:
        - query: Search query string
        - k: Number of top results to return

    user : UserContext
        Authenticated user context derived from JWT.

    Returns
    -------
    List[SearchResult]
        Ranked list of matching results.
    """
    # Embed the query
    query_embeddings = await embedder.embed([req.query])
    if not query_embeddings:
        return []

    query_embedding = query_embeddings[0]

    # Search using vector store
    results = await vector_store.search(
        wiki_id=user.wiki_id,
        query_embedding=query_embedding,
        k=req.k,
    )

    # Convert to response model
    return [
        SearchResult(
            title=title,
            score=score,
        )
        for title, section_id, namespace, score in results
    ]
