"""
FastAPI Dependency Injection Providers

This module defines reusable dependencies for routes, including:
- Database session
- LLM client singleton
- Vector store (PostgreSQL + pgvector)
- Embedder singleton
- Rate limiter
"""

from functools import lru_cache
from typing import Annotated, AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..llm.client import LLMClient
from ..db import get_async_session, VectorStore
from ..db.rate_limiter import RateLimiter
from ..embeddings.embedder import Embedder


@lru_cache
def get_llm_client() -> LLMClient:
    """Return a singleton LLM client instance."""
    return LLMClient()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session."""
    async for session in get_async_session():
        yield session


def get_vector_store(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> VectorStore:
    """
    Get a VectorStore instance with database session.
    
    The VectorStore uses PostgreSQL + pgvector for similarity search.
    """
    return VectorStore(session)


def get_rate_limiter(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> RateLimiter:
    """
    Get a RateLimiter instance with database session.
    
    The RateLimiter tracks and enforces daily token usage limits.
    """
    return RateLimiter(session)


@lru_cache
def get_embedder() -> Embedder:
    """Return a singleton embedder instance."""
    return Embedder()

