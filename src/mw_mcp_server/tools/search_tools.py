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

logger = logging.getLogger("mcp.search")

# Over-query multiplier: request N times more results from pgvector
# than needed to allow for post-filtering by page-level permissions.
VECTOR_OVERQUERY_MULTIPLIER = 3


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

async def tool_vector_search(
    query: str,
    user: UserContext,
    vector_store: VectorStore,
    embedder: Embedder,
    k: int = 5,
    client: Optional[MediaWikiClient] = None,
) -> List[ToolSearchResult]:
    """
    Semantic vector search tool callable by the LLM.

    Permission Filtering Pipeline:
    1. Over-query pgvector (request 3x results with namespace filter)
    2. Validate top results via MediaWiki API callback
    3. Return only accessible pages (up to k results)

    Parameters
    ----------
    query : str
        User query string.

    user : UserContext
        Authenticated MediaWiki user context.

    vector_store : VectorStore
        PostgreSQL + pgvector based vector store.

    embedder : Embedder
        Embedder capable of producing dense embeddings.

    k : int
        Maximum number of results to return.

    client : MediaWikiClient
        Optional client override for testing.

    Returns
    -------
    List[ToolSearchResult]
        Filtered, ranked results with text snippets.
    """
    # -------------------------------------------------------------
    # Early deny: empty allowed_namespaces means no access
    # -------------------------------------------------------------
    if not user.allowed_namespaces:
        return []

    # -------------------------------------------------------------
    # Embed the query text
    # -------------------------------------------------------------
    embeddings = await embedder.embed([query])
    if not embeddings:
        return []

    q_emb = embeddings[0]

    # -------------------------------------------------------------
    # Search with namespace pre-filtering
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # Validate page access via API callback (top 2x results)
    # -------------------------------------------------------------
    candidates = raw_results[:k * VECTOR_OVERQUERY_MULTIPLIER]
    titles_to_check = list(set(title for title, _, _, _ in candidates))

    try:
        access_map = await validate_page_access(titles_to_check, user, client)
    except Exception as exc:
        logger.error("Permission validation failed: %s", exc)
        raise ValueError(
            f"Permission validation failed during vector search: {type(exc).__name__}: {exc}"
        ) from exc

    # -------------------------------------------------------------
    # Filter to accessible pages and convert to API result objects
    # -------------------------------------------------------------
    results: List[ToolSearchResult] = []
    seen_titles = set()

    for title, section_id, namespace, score in candidates:
        if not access_map.get(title, False):
            continue

        # Deduplicate by title (keep highest score)
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

        # Stop once we have enough results
        if len(results) >= k:
            break

    return results


# ---------------------------------------------------------------------
# Standard Keyword Search Tool
# ---------------------------------------------------------------------

async def tool_search_pages(
    query: str,
    limit: int = 10,
    client: Optional[MediaWikiClient] = None,
    user: Optional[UserContext] = None,
) -> List[Dict[str, Any]]:
    """
    Perform a standard MediaWiki keyword search/list=search.

    Parameters
    ----------
    query : str
        Keyword search query.
    
    limit : int
        Max results.

    client : MediaWikiClient
        Optional client override.

    Returns
    -------
    List[Dict[str, Any]]
        List of search results with keys: title, snippet, size, wordcount, etc.
    """
    client = client or mw_client
    return await client.search_pages(query, limit=limit, user=user)
