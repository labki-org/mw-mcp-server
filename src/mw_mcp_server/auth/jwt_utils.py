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
"""

from __future__ import annotations

import jwt
import time
from typing import List, Dict, Any

from ..config import settings


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
    if not settings.jwt_mcp_to_mw_secret:
        raise JWTConfigurationError(
            "jwt_mcp_to_mw_secret is not configured. Cannot generate JWT."
        )

    if settings.jwt_ttl_seconds <= 0:
        raise JWTConfigurationError(
            f"JWT_TTL must be a positive integer; got {settings.jwt_ttl_seconds}"
        )


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def create_mcp_to_mw_jwt(scopes: List[str]) -> str:
    """
    Generate a short-lived JWT for MCP → MediaWiki communication.

    Parameters
    ----------
    scopes : List[str]
        List of granted scopes. Example: ["page_read", "page_write", "smw_query"]

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
        # NOTE: Additional claims like jti could be added for replay protection
    }

    # Signing key and algorithm
    secret = settings.jwt_mcp_to_mw_secret.get_secret_value()
    algo = settings.jwt_algo

    try:
        token = jwt.encode(payload, secret, algorithm=algo)
    except Exception as exc:
        raise JWTConfigurationError(
            f"Failed to generate JWT: {type(exc).__name__}: {str(exc)}"
        ) from exc

    return token
