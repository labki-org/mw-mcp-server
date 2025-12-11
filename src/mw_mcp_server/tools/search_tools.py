"""
Vector Search Tool

This module implements the LLM tool `mw_vector_search`, which performs
semantic search against the FAISS embedding index.

Responsibilities
----------------
- Embed the query
- Search FAISS
- Apply namespace‐based permission filtering
- Return structured ToolSearchResult objects
"""

from __future__ import annotations

from typing import List, Tuple

from ..embeddings.embedder import Embedder
from ..embeddings.index import FaissIndex
from ..auth.models import UserContext
from ..config import settings
from ..api.models import ToolSearchResult


# ---------------------------------------------------------------------
# Namespace Resolution
# ---------------------------------------------------------------------

def _resolve_allowed_namespaces_for(user: UserContext) -> List[int]:
    """
    Return the list of namespace IDs the current user is allowed to search in.

    Reads a comma‐separated list from configuration, e.g. "0,14".

    NOTE:
        This is a simple permission model today, but the structure
        allows extension to per-role or per-scope rules later.
    """
    raw = settings.allowed_namespaces_public
    if not raw:
        return []

    allowed = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            allowed.append(int(part))

    return allowed


# ---------------------------------------------------------------------
# Namespace Filtering
# ---------------------------------------------------------------------

def filter_by_permissions(
    results: List[Tuple[Any, float]],
    user_ctx: UserContext,
) -> List[Tuple[Any, float]]:
    """
    Filter FAISS search results based on namespace permissions.

    Parameters
    ----------
    results : List[(IndexedDocument, float)]
        Raw FAISS results.

    user_ctx : UserContext
        The authenticated user performing the search.

    Returns
    -------
    List[(IndexedDocument, float)]
        Only the results in allowed namespaces.
    """
    allowed_ns = _resolve_allowed_namespaces_for(user_ctx)
    if not allowed_ns:
        return []

    return [
        (doc, score)
        for (doc, score) in results
        if getattr(doc, "namespace", None) in allowed_ns
    ]


# ---------------------------------------------------------------------
# Main Search Tool
# ---------------------------------------------------------------------

async def tool_vector_search(
    query: str,
    user: UserContext,
    faiss_index: FaissIndex,
    embedder: Embedder,
    k: int = 5,
) -> List[ToolSearchResult]:
    """
    Semantic vector search tool callable by the LLM.

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

    Returns
    -------
    List[ToolSearchResult]
        Filtered, ranked results with text snippets.
    """

    # -------------------------------------------------------------
    # Validate FAISS index availability
    # -------------------------------------------------------------
    if (
        not getattr(faiss_index, "index", None)
        or faiss_index.index.ntotal == 0
    ):
        return []

    # -------------------------------------------------------------
    # Embed the query text
    # -------------------------------------------------------------
    embeddings = await embedder.embed([query])
    if not embeddings:
        return []

    q_emb = embeddings[0]

    # -------------------------------------------------------------
    # Run the FAISS search
    # -------------------------------------------------------------
    try:
        raw_results = faiss_index.search(q_emb, k)
    except Exception as exc:
        # The LLM tool protocol expects clean, structured errors.
        raise ValueError(f"FAISS search failed: {type(exc).__name__}") from exc

    # raw_results is expected to be List[(IndexedDocument, float)]
    if not raw_results:
        return []

    # -------------------------------------------------------------
    # Apply permission filtering
    # -------------------------------------------------------------
    permitted = filter_by_permissions(raw_results, user)

    # -------------------------------------------------------------
    # Convert to API result objects
    # -------------------------------------------------------------
    results: List[ToolSearchResult] = []

    for doc, score in permitted:
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
            # Defensive, but this should rarely trigger unless Docs are corrupt.
            raise ValueError(
                f"Failed to serialize search result: {type(exc).__name__}"
            ) from exc

    return results
