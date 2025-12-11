"""
Authentication Models

This module defines strongly-typed authentication and authorization models
used throughout the MCP server after JWT verification.
"""

from typing import List
from pydantic import BaseModel, Field, ConfigDict


class UserContext(BaseModel):
    """
    Authenticated user context derived from a verified JWT.

    This object is injected into all protected routes and represents the
    security boundary between MediaWiki and the MCP server.
    """

    username: str = Field(
        ...,
        min_length=1,
        description="MediaWiki username associated with the request.",
    )

    roles: List[str] = Field(
        default_factory=list,
        description="List of MediaWiki user groups (roles).",
    )

    scopes: List[str] = Field(
        default_factory=list,
        description="List of scopes granted to the user for API access.",
    )

    client_id: str = Field(
        ...,
        min_length=1,
        description="Client identifier that issued the JWT (e.g., MWAssistant).",
    )

    model_config = ConfigDict(
        frozen=True,                # Makes UserContext immutable after creation
        arbitrary_types_allowed=False,
        extra="forbid",             # Prevents claim injection via unexpected fields
    )
