"""
MCP Server Application Entry Point

This module defines the FastAPI application instance, registers all routers,
configures global exception handling, and provides a test-friendly application
factory.

Design Goals
------------
- Deterministic startup
- Explicit dependency initialization order
- Centralized router registration
- Global exception safety net
- Test-friendly via create_app()
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from fastapi import FastAPI
from sqlalchemy import text

from .api.dependencies import get_embedder, get_llm_client
from .config import settings
from .core.errors import unhandled_exception_handler
from .core.middleware import RequestIDLogFilter, RequestIDMiddleware
from .db import Base, async_engine
from .embeddings.queue import process_embeddings_worker_task

from .api import (
    chat_routes,
    search_routes,
    smw_routes,
    health_routes,
    embedding_routes,
    stats_routes,
)


# ---------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------

def _configure_logging() -> None:
    """
    Configure structured logging for production.
    Includes request_id in log output for distributed tracing.
    """
    log_level = settings.log_level.upper()

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | req=%(request_id)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )

    # Add the request ID filter to every handler on the root logger so that
    # all log records (including from third-party libraries like uvicorn)
    # have the request_id attribute populated before formatting.
    request_id_filter = RequestIDLogFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(request_id_filter)


_configure_logging()
logger = logging.getLogger("mcp.app")


# ---------------------------------------------------------------------
# Application Lifespan (Startup/Shutdown)
# ---------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application lifecycle events.

    This replaces the deprecated @app.on_event() decorators with
    the modern lifespan context manager pattern.

    Startup:
    - Validates critical configuration
    - Initializes database connection and creates tables
    - Enables pgvector extension

    Shutdown:
    - Disposes database connection pool
    """
    # ---- Startup ----
    logger.info("Starting mw-mcp-server")

    # Fail-fast: validate critical secrets at startup
    try:
        _ = settings.openai_api_key.get_secret_value()

        logger.info("Configuration validated successfully")
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise

    # Initialize database
    try:
        async with async_engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables initialized")

        alembic_cfg = AlembicConfig("alembic.ini")
        await asyncio.to_thread(alembic_command.upgrade, alembic_cfg, "head")
        logger.info("Database migrations completed")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

    worker_task = asyncio.create_task(process_embeddings_worker_task())
    logger.info("Background embedding worker started")

    yield  # Application runs here

    # ---- Shutdown ----
    logger.info("Shutting down mw-mcp-server")

    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        logger.info("Background embedding worker cancelled cleanly")

    # Close long-lived HTTP clients on the cached singletons.
    await get_llm_client().aclose()
    await get_embedder().aclose()

    await async_engine.dispose()
    logger.info("Database connections closed")


# ---------------------------------------------------------------------
# Application Factory (Test-Friendly)
# ---------------------------------------------------------------------

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    This factory pattern allows:
    - Clean test instantiation
    - Isolated app instances for integration tests
    - Controlled dependency overrides in pytest

    Returns
    -------
    FastAPI
        Fully configured FastAPI application.
    """
    app = FastAPI(
        title="mw-mcp-server",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # --------------------------------------------------------------
    # Middleware (order matters: first added = outermost)
    # --------------------------------------------------------------

    app.add_middleware(RequestIDMiddleware)

    # --------------------------------------------------------------
    # Global Exception Handling
    # --------------------------------------------------------------

    app.add_exception_handler(Exception, unhandled_exception_handler)

    # --------------------------------------------------------------
    # Router Registration
    # --------------------------------------------------------------

    app.include_router(health_routes.router)
    app.include_router(chat_routes.router)
    app.include_router(search_routes.router)
    app.include_router(smw_routes.router)
    app.include_router(embedding_routes.router)
    app.include_router(stats_routes.router)

    return app


# ---------------------------------------------------------------------
# Default Application Instance (for Uvicorn/Gunicorn)
# ---------------------------------------------------------------------

app = create_app()
