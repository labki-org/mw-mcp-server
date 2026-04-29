"""
Vector Search Tool

This module implements the LLM tool `mw_vector_search`, which performs
semantic search against the PostgreSQL + pgvector embedding index.

Responsibilities
----------------
- Embed the query
- Search pgvector
- Apply namespace-based permission pre-filtering (from JWT)
- Validate page-level access via MediaWiki API callback
- Return structured ToolSearchResult objects
"""

from __future__ import annotations

import logging
from typing import List, Optional, Any, Dict

from ..wiki.api_client import MediaWikiClient
from .wiki_tools import mw_client

from ..embeddings.embedder import Embedder
from ..db import VectorStore
from ..auth.models import UserContext
from ..api.models import ToolSearchResult
from .pagination import paginated

logger = logging.getLogger("mcp.search")

# Over-query multiplier: request N times more results from pgvector
# than needed to allow for post-filtering by page-level permissions.
VECTOR_OVERQUERY_MULTIPLIER = 3

# How many of the over-queried results to validate via MediaWiki API.
# Lower than VECTOR_OVERQUERY_MULTIPLIER to reduce permission-check API calls.
PERMISSION_CHECK_MULTIPLIER = 2


# ---------------------------------------------------------------------
# Permission Validation via API Callback
# ---------------------------------------------------------------------

async def validate_page_access(
    titles: List[str],
    user: UserContext,
    client: Optional[MediaWikiClient] = None,
) -> Dict[str, bool]:
    """
    Validate that the user can read each page via MediaWiki API callback.

    This catches ControlAccess page-level restrictions that aren't covered
    by namespace-level Lockdown pre-filtering.

    Parameters
    ----------
    titles : List[str]
        Page titles to validate.

    user : UserContext
        Authenticated user context.

    client : MediaWikiClient
        Optional client override for testing.

    Returns
    -------
    Dict[str, bool]
        Map of page title to access granted boolean.
    """
    if not titles:
        return {}

    client = client or mw_client
    return await client.check_read_access(titles, user)


# ---------------------------------------------------------------------
# Main Search Tool
# ---------------------------------------------------------------------

async def vector_search(
    query: str,
    user: UserContext,
    vector_store: VectorStore,
    embedder: Embedder,
    k: int = 5,
    client: Optional[MediaWikiClient] = None,
) -> List[ToolSearchResult]:
    """
    Run a permission-filtered vector search and return the raw result list.

    This is the lower-level building block used by both the HTTP search
    route and the LLM tool wrapper. The LLM-facing wrapper (``tool_vector_search``)
    adds truncation metadata; the HTTP route uses the bare list.
    """
    # Early deny: empty allowed_namespaces means no access
    if not user.allowed_namespaces:
        return []

    embeddings = await embedder.embed([query])
    if not embeddings:
        return []

    q_emb = embeddings[0]

    try:
        raw_results = await vector_store.search(
            wiki_id=user.wiki_id,
            query_embedding=q_emb,
            k=k * VECTOR_OVERQUERY_MULTIPLIER,  # Over-query to allow for filtering
            namespace_filter=user.allowed_namespaces,
        )
    except Exception as exc:
        logger.exception("Vector search failed")
        raise ValueError(f"Vector search failed: {type(exc).__name__}") from exc

    if not raw_results:
        return []

    candidates = raw_results[:k * PERMISSION_CHECK_MULTIPLIER]
    titles_to_check = list({title for title, _, _, _ in candidates})

    try:
        access_map = await validate_page_access(titles_to_check, user, client)
    except Exception as exc:
        logger.error("Permission validation failed: %s", exc)
        raise ValueError(
            f"Permission validation failed during vector search: {type(exc).__name__}: {exc}"
        ) from exc

    results: List[ToolSearchResult] = []
    seen_titles: set = set()

    for title, section_id, namespace, score in candidates:
        if not access_map.get(title, False):
            continue
        if title in seen_titles:
            continue
        seen_titles.add(title)

        results.append(
            ToolSearchResult(
                title=title,
                section_id=section_id,
                score=float(score),
            )
        )

        if len(results) >= k:
            break

    return results


async def tool_vector_search(
    query: str,
    user: UserContext,
    vector_store: VectorStore,
    embedder: Embedder,
    k: int = 5,
    client: Optional[MediaWikiClient] = None,
) -> Dict[str, Any]:
    """
    LLM-facing semantic vector search tool.

    Wraps :func:`vector_search` with truncation metadata so the LLM can
    detect when ``k`` capped the result and decide whether to widen it.
    Returns ``{results, count, limit, truncated, note}`` where each item
    is a ``ToolSearchResult``-shaped dict.
    """
    results = await vector_search(
        query=query,
        user=user,
        vector_store=vector_store,
        embedder=embedder,
        k=k,
        client=client,
    )
    return paginated([r.model_dump() for r in results], limit=k)


# ---------------------------------------------------------------------
# Standard Keyword Search Tool
# ---------------------------------------------------------------------

async def tool_search_pages(
    query: str,
    limit: int = 10,
    client: Optional[MediaWikiClient] = None,
    user: Optional[UserContext] = None,
) -> Dict[str, Any]:
    """
    Perform a standard MediaWiki keyword search/list=search.

    Returns a paginated wrapper ``{results, count, limit, truncated, note}``;
    each result is the raw MW search-result dict (title/snippet/size/etc.).
    """
    client = client or mw_client
    rows = await client.search_pages(query, limit=limit, user=user)
    return paginated(rows, limit=limit)
