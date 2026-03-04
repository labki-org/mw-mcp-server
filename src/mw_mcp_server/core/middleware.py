"""
Request Tracing Middleware

Extracts or generates a unique request ID for each request and injects
it into log records and response headers for distributed tracing.
"""

import logging
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Context variable for the current request ID, accessible anywhere in the call stack
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

logger = logging.getLogger("mcp.middleware")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that propagates X-Request-ID for distributed tracing."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Use incoming header if present, otherwise generate one
        req_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request_id_var.set(req_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response


class RequestIDLogFilter(logging.Filter):
    """Logging filter that injects the current request_id into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get("")
        return True
