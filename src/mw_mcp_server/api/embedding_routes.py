"""
Embeddings Routes

This module exposes endpoints for:
- Querying FAISS embedding index statistics
- Updating page embeddings
- Deleting page embeddings

These endpoints are designed to be invoked by trusted MediaWiki services
for synchronizing wiki page content into the MCP vector search layer.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Annotated

from .models import (
    EmbeddingUpdatePageRequest,
    EmbeddingDeletePageRequest,
    EmbeddingStatsResponse,
    OperationResult,
)
from .dependencies import get_faiss_index, get_embedder
from ..auth.security import require_scopes
from ..auth.models import UserContext
from ..embeddings.models import IndexedDocument
from ..embeddings.index import FaissIndex
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


def _build_documents(
    title: str,
    chunks: List[str],
    last_modified: str,
    namespace: int = 0,
) -> List[IndexedDocument]:
    """
    Build IndexedDocument records from text chunks.
    """
    return [
        IndexedDocument(
            page_title=title,
            text=chunk,
            namespace=namespace,
            last_modified=last_modified,
        )
        for chunk in chunks
    ]


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
    faiss_index: Annotated[FaissIndex, Depends(get_faiss_index)],
) -> EmbeddingStatsResponse:
    """
    Return current FAISS index statistics.
    """
    # Global exception handler captures failures
    return faiss_index.get_stats()


@router.post(
    "/page",
    summary="Create or update a page embedding",
    response_model=OperationResult,
)
async def update_page_embedding(
    req: EmbeddingUpdatePageRequest,
    user: Annotated[UserContext, Depends(require_scopes("embeddings"))],
    faiss_index: Annotated[FaissIndex, Depends(get_faiss_index)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> OperationResult:
    """
    Create or update embeddings for a wiki page.

    Workflow
    --------
    1. Chunk page content into paragraphs.
    2. Delete any existing embeddings for the page.
    3. Embed the new content.
    4. Add new vectors to the FAISS index.
    5. Persist the updated index.
    """

    # -------------------------------------------------------------
    # 1. Chunk Content
    # -------------------------------------------------------------
    text_chunks = _chunk_text(req.content)

    documents = _build_documents(
        title=req.title,
        chunks=text_chunks,
        last_modified=req.last_modified,
        namespace=0,
    )

    # -------------------------------------------------------------
    # 2. Delete Existing Page Embeddings
    # -------------------------------------------------------------
    faiss_index.delete_page(req.title)

    # If no valid content remains after chunking, persist deletion and exit early
    if not documents:
        faiss_index.save()

        return OperationResult(
            status="deleted",
            count=0,
            details={"reason": "empty_content_after_chunking"}
        )

    # -------------------------------------------------------------
    # 3. Embed & Add to Index
    # -------------------------------------------------------------
    # Global exception handler catches embedding/faiss errors
    embeddings = await embedder.embed(text_chunks)
    faiss_index.add_documents(documents, embeddings)
    faiss_index.save()

    return OperationResult(
        status="updated",
        count=len(documents)
    )


@router.delete(
    "/page",
    summary="Delete embeddings for a wiki page",
    response_model=OperationResult,
)
async def delete_page_embedding(
    req: EmbeddingDeletePageRequest,
    user: Annotated[UserContext, Depends(require_scopes("embeddings"))],
    faiss_index: Annotated[FaissIndex, Depends(get_faiss_index)],
) -> OperationResult:
    """
    Delete all embeddings for a given wiki page.
    """
    count = faiss_index.delete_page(req.title)
    faiss_index.save()

    return OperationResult(
        status="deleted",
        count=count
    )
