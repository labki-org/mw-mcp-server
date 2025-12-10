"""
Search Routes

This module defines vector-based semantic search endpoints backed by the
FAISS embedding index. These routes are typically invoked by the LLM during
tool execution as well as by the MediaWiki client for direct user queries.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Annotated

from .models import SearchRequest, SearchResult
from ..auth.security import require_scopes
from ..auth.models import UserContext
from ..tools.search_tools import tool_vector_search
from ..embeddings.index import FaissIndex
from ..embeddings.embedder import Embedder
from .dependencies import get_faiss_index, get_embedder

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
    faiss_index: Annotated[FaissIndex, Depends(get_faiss_index)],
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
    # The global exception handler 'unhandled_exception_handler' will catch
    # any exceptions raised by 'tool_vector_search', log them, and return a 500.
    
    raw_results = await tool_vector_search(
        query=req.query,
        user=user,
        faiss_index=faiss_index,
        embedder=embedder,
        k=req.k,
    )

    # Defensive schema validation of tool output is still useful for data integrity,
    # but we can let Pydantic validation errors propagate or handle them specifically if needed.
    # For now, we'll keep the transformation simple.
    return [SearchResult(**result) for result in raw_results]
