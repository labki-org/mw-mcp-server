from fastapi import Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger("mcp")

async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled MCP exception", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
