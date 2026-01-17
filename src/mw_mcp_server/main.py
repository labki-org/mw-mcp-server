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

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from sqlalchemy import text

from .config import settings
from .core.errors import unhandled_exception_handler
from .db import async_engine, Base

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
    """
    log_level = getattr(settings, "log_level", "INFO").upper()
    
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )


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
            # Enable pgvector extension
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

    yield  # Application runs here

    # ---- Shutdown ----
    logger.info("Shutting down mw-mcp-server")

    # Close database connections
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
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

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
