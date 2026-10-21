"""
Shared helpers for tests that need a real PostgreSQL + pgvector database.

Tests using the ``pg_session_factory`` fixture are skipped when DATABASE_URL
isn't reachable (CI provides a pgvector service container). Each test gets a
fresh NullPool engine bound to its own event loop and empty tables.
"""

import asyncio

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from mw_mcp_server.config import settings
from mw_mcp_server.db.models import Base

_TABLES = "embedding_job, embedding, chat_message, chat_session, token_usage"


@pytest_asyncio.fixture(name="pg_session_factory")
async def pg_session_factory_fixture():
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    try:
        async with asyncio.timeout(3):
            async with engine.begin() as conn:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await conn.run_sync(Base.metadata.create_all)
    except Exception as exc:  # pragma: no cover - environment dependent
        await engine.dispose()
        pytest.skip(f"PostgreSQL not available: {type(exc).__name__}")

    async with engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE {_TABLES} RESTART IDENTITY CASCADE"))

    yield async_sessionmaker(engine, expire_on_commit=False, autoflush=False)

    await engine.dispose()
