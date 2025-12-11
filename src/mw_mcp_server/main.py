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
from fastapi import FastAPI

from .config import settings
from .core.errors import unhandled_exception_handler

from .api import (
    chat_routes,
    search_routes,
    smw_routes,
    health_routes,
    embedding_routes,
)


logger = logging.getLogger("mcp.app")


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

    # --------------------------------------------------------------
    # Startup Validation Hook
    # --------------------------------------------------------------

    @app.on_event("startup")
    async def _startup_validation() -> None:
        """
        Fail-fast validation at application startup.

        This ensures that critical configuration is present before the first
        request is ever served.
        """
        logger.info("Starting mw-mcp-server")

        # Touch critical secrets to force validation now (not at first use)
        _ = settings.openai_api_key.get_secret_value()
        _ = settings.jwt_mw_to_mcp_secret.get_secret_value()
        _ = settings.jwt_mcp_to_mw_secret.get_secret_value()

        logger.info("Configuration validated successfully")

    # --------------------------------------------------------------
    # Shutdown Hook (Future-Proofing)
    # --------------------------------------------------------------

    @app.on_event("shutdown")
    async def _shutdown_cleanup() -> None:
        """
        Graceful shutdown hook.

        This is the correct place to:
        - Close HTTP clients
        - Flush FAISS indexes
        - Terminate background workers
        """
        logger.info("Shutting down mw-mcp-server")

    return app


# ---------------------------------------------------------------------
# Default Application Instance (for Uvicorn/Gunicorn)
# ---------------------------------------------------------------------

app = create_app()
