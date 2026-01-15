"""
FastAPI Dependency Injection Providers

This module defines reusable dependencies for routes, including:
- LLM client singleton
- Tenant-aware FAISS index lookup
- Embedder singleton
"""

from functools import lru_cache

from fastapi import Depends

from ..llm.client import LLMClient
from ..embeddings.index import FaissIndex
from ..embeddings.embedder import Embedder
from ..embeddings.registry import get_tenant_index
from ..auth.security import verify_mw_to_mcp_jwt
from ..auth.models import UserContext


@lru_cache
def get_llm_client() -> LLMClient:
    """Return a singleton LLM client instance."""
    return LLMClient()


def get_tenant_faiss_index(
    user: UserContext = Depends(verify_mw_to_mcp_jwt),
) -> FaissIndex:
    """
    Get the FAISS index for the authenticated user's wiki.

    This function uses the wiki_id from the verified JWT to return
    a tenant-isolated index instance.
    """
    return get_tenant_index(user.wiki_id)


# Backwards compatibility alias - routes should migrate to get_tenant_faiss_index
def get_faiss_index(
    user: UserContext = Depends(verify_mw_to_mcp_jwt),
) -> FaissIndex:
    """
    Deprecated: Use get_tenant_faiss_index instead.
    
    This alias maintains backwards compatibility while we migrate routes.
    """
    return get_tenant_index(user.wiki_id)


@lru_cache
def get_embedder() -> Embedder:
    """Return a singleton embedder instance."""
    return Embedder()

