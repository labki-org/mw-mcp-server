"""
Vector Search Tool

This module implements the LLM tool `mw_vector_search`, which performs
semantic search against the FAISS embedding index.

Responsibilities
----------------
- Embed the query
- Search FAISS
- Apply namespace-based permission pre-filtering (from JWT)
- Validate page-level access via MediaWiki API callback
- Return structured ToolSearchResult objects
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional, Any, Dict

from ..wiki.api_client import MediaWikiClient
from .wiki_tools import mw_client

from ..embeddings.embedder import Embedder
from ..embeddings.index import FaissIndex
from ..auth.models import UserContext
from ..api.models import ToolSearchResult

logger = logging.getLogger("mcp.search")


# ---------------------------------------------------------------------
# Namespace Pre-Filtering
# ---------------------------------------------------------------------

def filter_by_namespace(
    results: List[Tuple[Any, float]],
    allowed_namespaces: List[int],
) -> List[Tuple[Any, float]]:
    """
    Pre-filter FAISS results by the user's allowed namespaces.

    This is a fast first-pass filter using Lockdown-derived namespace
    permissions included in the JWT.

    Parameters
    ----------
    results : List[(IndexedDocument, float)]
        Raw FAISS results.

    allowed_namespaces : List[int]
        Namespace IDs the user can read (from JWT).

    Returns
    -------
    List[(IndexedDocument, float)]
        Only the results in allowed namespaces.
    """
    if not allowed_namespaces:
        # If no allowed namespaces specified, deny all (safe default)
        logger.warning("No allowed namespaces in user context, denying all results")
        return []

    return [
        (doc, score)
        for (doc, score) in results
        if getattr(doc, "namespace", None) in allowed_namespaces
    ]


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
    return await client.check_read_access(titles, user.username)


# ---------------------------------------------------------------------
# Main Search Tool
# ---------------------------------------------------------------------

async def tool_vector_search(
    query: str,
    user: UserContext,
    faiss_index: FaissIndex,
    embedder: Embedder,
    k: int = 5,
    client: Optional[MediaWikiClient] = None,
) -> List[ToolSearchResult]:
    """
    Semantic vector search tool callable by the LLM.

    Permission Filtering Pipeline:
    1. Over-query FAISS (request 3x results)
    2. Pre-filter by user's allowed namespaces (from JWT)
    3. Validate top results via MediaWiki API callback
    4. Return only accessible pages (up to k results)

    Parameters
    ----------
    query : str
        User query string.

    user : UserContext
        Authenticated MediaWiki user context.

    faiss_index : FaissIndex
        Active FAISS vector index.

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
    # Embed the query text
    # -------------------------------------------------------------
    embeddings = await embedder.embed([query])
    if not embeddings:
        return []

    q_emb = embeddings[0]

    # -------------------------------------------------------------
    # Over-query FAISS (3x requested to allow for filtering)
    # -------------------------------------------------------------
    try:
        raw_results = faiss_index.search(q_emb, k * 3)
    except Exception as exc:
        raise ValueError(f"FAISS search failed: {type(exc).__name__}") from exc

    if not raw_results:
        return []

    # -------------------------------------------------------------
    # Pre-filter by namespace permissions (from JWT)
    # -------------------------------------------------------------
    ns_filtered = filter_by_namespace(raw_results, user.allowed_namespaces)

    if not ns_filtered:
        logger.info("No results after namespace filtering for user %s", user.username)
        return []

    # -------------------------------------------------------------
    # Validate page access via API callback (top 2x results)
    # -------------------------------------------------------------
    # Request more than k to ensure we have enough after page-level filtering
    candidates = ns_filtered[:k * 2]
    titles_to_check = [doc.page_title for doc, _ in candidates]

    try:
        access_map = await validate_page_access(titles_to_check, user, client)
    except Exception as exc:
        logger.error("Permission validation failed: %s", exc)
        # Safe default: deny all if permission check fails
        return []

    # -------------------------------------------------------------
    # Filter to accessible pages and convert to API result objects
    # -------------------------------------------------------------
    results: List[ToolSearchResult] = []

    for doc, score in candidates:
        if not access_map.get(doc.page_title, False):
            continue

        try:
            results.append(
                ToolSearchResult(
                    title=doc.page_title,
                    section_id=getattr(doc, "section_id", None),
                    score=float(score),
                    text=(doc.text[:400] if getattr(doc, "text", None) else ""),
                )
            )
        except Exception as exc:
            raise ValueError(
                f"Failed to serialize search result: {type(exc).__name__}"
            ) from exc

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
    
    # We must ensure we have a client instance since mw_client might be initialized without async loop
    # actually mw_client is global at module level.
    
    return await client.search_pages(query, limit=limit, user=user)
