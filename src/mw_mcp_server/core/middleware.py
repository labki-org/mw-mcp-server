"""
Request Tracing Middleware

Extracts or generates a unique request ID for each request and injects
it into log records and response headers for distributed tracing.
"""

import logging
import re
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Context variable for the current request ID, accessible anywhere in the call stack
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

logger = logging.getLogger("mcp.middleware")

# Client-supplied IDs are echoed into logs and response headers, so only
# accept a conservative charset/length (prevents log forging and header abuse).
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def resolve_request_id(incoming: str | None) -> str:
    """Return ``incoming`` if it is a well-formed request ID, else a fresh one."""
    if incoming and _REQUEST_ID_RE.fullmatch(incoming):
        return incoming
    return uuid.uuid4().hex[:16]


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that propagates X-Request-ID for distributed tracing."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Use a well-formed incoming header if present, otherwise generate one
        req_id = resolve_request_id(request.headers.get("x-request-id"))
        request_id_var.set(req_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response


class RequestIDLogFilter(logging.Filter):
    """Logging filter that injects the current request_id into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get("")
        return True
