"""
Actions Routes: Edit Page Endpoint

This module defines action-oriented API routes that are invoked by authorized
MediaWiki clients through the MCP server.

Current Responsibilities:
- Receive edit requests from authenticated clients
- Enforce scope-based authorization
- Delegate edit execution to the tool layer
- Return structured, validated responses

Security Model:
- Scope-based access control via `require_scopes`
- Full user identity passed via `UserContext`
- No direct MediaWiki credentials handled here
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated

from .models import EditRequest, OperationResult
from ..auth.security import require_scopes
from ..auth.models import UserContext
from ..tools.edit_tools import tool_apply_edit

# ---------------------------------------------------------------------
# Router Configuration
# ---------------------------------------------------------------------

router = APIRouter(
    prefix="/actions",
    tags=["actions"],
)

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@router.post(
    "/edit-page",
    response_model=OperationResult,
    status_code=status.HTTP_200_OK,
    summary="Apply an edit to a MediaWiki page",
    description=(
        "Applies a text update to a MediaWiki page on behalf of an authenticated "
        "user. Requires the `edit_page` scope."
    ),
)
async def edit_page(
    req: EditRequest,
    user: Annotated[UserContext, Depends(require_scopes("edit_page"))],
) -> OperationResult:
    """
    Apply an edit to a MediaWiki page.

    This endpoint is invoked by the MediaWiki extension after an LLM proposes
    an edit and the user authorizes execution.

    Parameters
    ----------
    req : EditRequest
        The edit request payload containing:
        - Page title
        - New page content
        - Edit summary

    user : UserContext
        Authenticated user context extracted from a verified JWT token.

    Returns
    -------
    OperationResult
        Structured result of the edit operation.
    """
    # Global exception handler captures any tool failures
    result = await tool_apply_edit(
        title=req.title,
        new_text=req.new_text,
        summary=req.summary,
        user=user,
    )

    # Transform internal tool result to standardized OperationResult
    return OperationResult(
        status="updated",
        details=result
    )
