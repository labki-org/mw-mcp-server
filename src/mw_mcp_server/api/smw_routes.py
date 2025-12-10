"""
Semantic MediaWiki Query Routes

This module exposes an internal endpoint for executing SMW `ask` queries
via the MCP server. These queries are intended to be issued by trusted
MediaWiki extension components or LLM tools.

Security Notes
--------------
- Requires the `smw_query` scope.
- Queries are NOT sanitized here because the MediaWiki-side SMW API
  enforces its own constraints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated, Dict, Any

from .models import SMWQueryRequest, SMWQueryResponse
from ..auth.security import require_scopes
from ..auth.models import UserContext
from ..tools.wiki_tools import tool_run_smw_ask

router = APIRouter(prefix="/smw-query", tags=["smw"])


@router.post(
    "/",
    response_model=SMWQueryResponse,
    summary="Execute a Semantic MediaWiki ASK query",
    status_code=status.HTTP_200_OK,
)
async def smw_query(
    req: SMWQueryRequest,
    user: Annotated[UserContext, Depends(require_scopes("smw_query"))],
) -> SMWQueryResponse:
    """
    Execute an SMW ASK query through the MediaWiki API.

    Parameters
    ----------
    req : SMWQueryRequest
        Contains a single `ask` field with the raw SMW query string.
    user : UserContext
        Authenticated user identity extracted by JWT + scope checks.

    Returns
    -------
    SMWQueryResponse
        Raw SMW query results.

    Raises
    ------
    HTTPException
        - 500 if the tool fails or returns invalid data.
    """
    # Global exception handler captures any tool failures
    result: Dict[str, Any] = await tool_run_smw_ask(req.ask, user)

    # Validate that result is at least a dict-like object
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid SMW query result returned by backend.",
        )

    return SMWQueryResponse(raw=result)
