"""
Health Routes

This module exposes lightweight health and readiness endpoints for the MCP
server. These endpoints are intended for:
- Load balancers
- Kubernetes probes
- Deployment sanity checks
- Basic runtime diagnostics

They MUST remain:
- Fast
- Side-effect free
- Free of secrets
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .dependencies import get_db_session

router = APIRouter(tags=["health"])

logger = logging.getLogger("mcp.health")


@router.get(
    "/health",
    summary="Service liveness check",
    status_code=status.HTTP_200_OK,
)
def health() -> Dict[str, str]:
    """
    Basic liveness probe. Returns 200 if the process is running.
    """
    return {
        "status": "ok",
    }


@router.get(
    "/health/ready",
    summary="Service readiness check",
)
async def health_ready(
    session: AsyncSession = Depends(get_db_session),
) -> JSONResponse:
    """
    Deep readiness probe. Verifies:
    - Database connectivity (SELECT 1)

    Returns 200 if all checks pass, 503 otherwise.
    """
    checks: Dict[str, Any] = {}

    # Database check
    try:
        await session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        logger.error("Health check: database unreachable: %s", exc)
        checks["database"] = "error"

    all_ok = all(v == "ok" for v in checks.values())

    return JSONResponse(
        status_code=status.HTTP_200_OK if all_ok else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "ok" if all_ok else "degraded",
            "checks": checks,
        },
    )
