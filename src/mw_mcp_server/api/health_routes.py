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

from fastapi import APIRouter, status
from typing import Dict


router = APIRouter(tags=["health"])


@router.get(
    "/health",
    summary="Service health check",
    status_code=status.HTTP_200_OK,
)
def health() -> Dict[str, str]:
    """
    Basic liveness probe.

    Returns
    -------
    Dict[str, str]
        Minimal runtime status and MediaWiki API endpoint reference.
        The MW API URL is returned only as a string for debug visibility
        and contains no credentials.
    """
    return {
        "status": "ok",
    }
