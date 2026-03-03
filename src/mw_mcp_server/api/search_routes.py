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
    # Early deny: empty allowed_namespaces means no access
    if not user.allowed_namespaces:
        return []

    # Embed the query
    query_embeddings = await embedder.embed([req.query])
    if not query_embeddings:
        return []

    query_embedding = query_embeddings[0]

    # Over-query to allow for permission filtering
    k = req.k
    results = await vector_store.search(
        wiki_id=user.wiki_id,
        query_embedding=query_embedding,
        k=k * 3,
        namespace_filter=user.allowed_namespaces,
    )

    if not results:
        return []

    # Validate page-level access via MediaWiki API callback
    from ..tools.search_tools import validate_page_access

    candidates = results[:k * 2]
    titles_to_check = list(set(title for title, _, _, _ in candidates))

    try:
        access_map = await validate_page_access(titles_to_check, user)
    except Exception:
        # Safe default: deny all if permission check fails
        return []

    # Filter to accessible pages, deduplicate, return top k
    seen_titles: set = set()
    filtered: list = []
    for title, section_id, namespace, score in candidates:
        if not access_map.get(title, False):
            continue
        if title in seen_titles:
            continue
        seen_titles.add(title)
        filtered.append(
            SearchResult(
                title=title,
                score=score,
            )
        )
        if len(filtered) >= k:
            break

    return filtered
