"""
Global Error Handling

This module defines application-wide exception handlers for the MCP server.

Design Goals
------------
- Never leak internal exception details to clients
- Always return deterministic, machine-readable error responses
- Log full stack traces internally for debugging
- Remain testable and framework-agnostic where possible
"""

from __future__ import annotations

import logging
from typing import Dict, Any

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("mcp.errors")


# ---------------------------------------------------------------------
# Public Exception Handlers
# ---------------------------------------------------------------------

async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Catch-all handler for uncaught exceptions.

    This handler should be registered with FastAPI as the final safety net
    for any exception not otherwise handled by route-level or framework-level
    handlers.

    Behavior
    --------
    - Logs the full exception stack trace for internal diagnostics.
    - Returns a generic 500 error to the client with no internal details.
    - Ensures consistent error response format across the entire API.

    Parameters
    ----------
    request : Request
        The incoming HTTP request that triggered the exception.

    exc : Exception
        The uncaught exception instance.

    Returns
    -------
    JSONResponse
        A JSON 500 response with a minimal error payload.
    """

    # Log full traceback internally (never returned to client)
    logger.exception(
        "Unhandled MCP exception during request: %s %s",
        request.method,
        request.url.path,
        exc_info=exc,
    )

    # Deterministic, minimal external error surface
    payload: Dict[str, Any] = {
        "error": "internal_server_error",
        "detail": "Internal server error",
    }

    return JSONResponse(
        status_code=500,
        content=payload,
    )
