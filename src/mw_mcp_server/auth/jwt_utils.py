"""
JWT Utility Functions

This module provides helpers for generating short-lived JWT tokens used for
secure MCP → MediaWiki communication. These JWTs are *not* user tokens; they
are strictly server-to-server authentication credentials.

Key characteristics:
- Short-lived (TTL configured in settings)
- Scoped
- Signed with a dedicated MCP→MW secret
- Includes explicit issuer/audience claims
- Bound to the target wiki (``wiki_id``) and, when acting for a user, to that
  user (``user`` / ``user_id``), so MWAssistant can reject requests whose
  username/user_id params do not match the signed identity
- Carries a unique ``jti``
"""

from __future__ import annotations

import jwt
import time
import uuid
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from ..config import settings

if TYPE_CHECKING:
    from .models import UserContext


# ---------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------

class JWTConfigurationError(RuntimeError):
    """Raised when JWT generation cannot proceed due to configuration issues."""


# ---------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------

def _get_current_timestamp() -> int:
    """Return current UNIX timestamp in UTC as integer seconds."""
    return int(time.time())


def _validate_jwt_config() -> None:
    """
    Ensures required JWT configuration is present.
    Raises a structured exception instead of failing deep inside jwt.encode().
    """
    if not settings.wiki_creds:
        raise JWTConfigurationError(
            "wiki_creds are not configured. Cannot generate JWT."
        )

    if settings.jwt_ttl_seconds <= 0:
        raise JWTConfigurationError(
            f"JWT_TTL must be a positive integer; got {settings.jwt_ttl_seconds}"
        )


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def create_mcp_to_mw_jwt(
    scopes: List[str],
    wiki_id: Optional[str] = None,
    user: Optional["UserContext"] = None,
) -> str:
    """
    Generate a short-lived JWT for MCP → MediaWiki communication.

    Parameters
    ----------
    scopes : List[str]
        List of granted scopes. Example: ["page_read", "page_write", "smw_query"]
        
    wiki_id : Optional[str]
        ID of the target wiki. If provided, uses the specific secret for that wiki.
        Always included as the ``wiki_id`` claim.

    user : Optional[UserContext]
        Authenticated user the request is made on behalf of. When given, the
        token carries signed ``user`` and ``user_id`` claims, which
        MWAssistant uses as the acting identity. Omit for system calls
        (e.g. embedding jobs) that do not act for a user.

    Returns
    -------
    str
        Encoded JWT suitable for use in Authorization: Bearer <token> header.

    Raises
    ------
    JWTConfigurationError
        If configuration is missing or invalid.
    """
    _validate_jwt_config()

    now = _get_current_timestamp()

    # Construct payload with required claims.
    payload: Dict[str, Any] = {
        "iss": "mw-mcp-server",
        "aud": "MWAssistant",           # Must match the MW extension's expected audience
        "iat": now,                    # Issued at
        "exp": now + settings.jwt_ttl_seconds, # Short TTL for safety
        "scope": scopes,               # Scope-based capabilities
        "jti": uuid.uuid4().hex,       # Unique token ID
    }
    if wiki_id:
        payload["wiki_id"] = wiki_id

    if user is not None:
        # MWAssistant treats user_id 0 as anonymous and otherwise requires
        # the name and id to resolve to the same account.
        payload["user_id"] = int(user.user_id)
        if user.user_id and user.username:
            payload["user"] = user.username

    # Select signing secret
    secret = None
    if user is not None and wiki_id and user.wiki_id != wiki_id:
        raise JWTConfigurationError(
            "Refusing to sign a token for a user of a different wiki."
        )

    if wiki_id and wiki_id in settings.wiki_creds:
        secret = settings.wiki_creds[wiki_id].mcp_to_mw_secret.get_secret_value()
        
    if not secret:
        raise JWTConfigurationError(
            f"No signing secret available for wiki_id='{wiki_id}'."
        )

    # Signing key and algorithm
    algo = settings.jwt_algo

    try:
        token = jwt.encode(payload, secret, algorithm=algo)
    except Exception as exc:
        raise JWTConfigurationError(
            f"Failed to generate JWT: {type(exc).__name__}: {str(exc)}"
        ) from exc

    return token
