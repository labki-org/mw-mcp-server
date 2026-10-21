"""
JWT Verification & Scope Enforcement

This module is responsible for:

1. Verifying incoming JWTs issued by the MWAssistant MediaWiki extension.
2. Enforcing scope-based authorization rules.
3. Producing a validated `UserContext` object to downstream routes.

Security Model
--------------
- Incoming JWTs use a *different secret* from outbound MCP→MW JWTs.
- Incoming JWTs represent *user identity and permissions*.
- They must be short-lived and include issuer, audience, and scope claims.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Set
from urllib.parse import urlparse

import jwt

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..config import settings
from .models import UserContext

logger = logging.getLogger("mcp.auth")


# ---------------------------------------------------------------------
# Security Scheme
# ---------------------------------------------------------------------

security = HTTPBearer(auto_error=True)


# ---------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------

class JWTVerificationError(RuntimeError):
    """Raised internally when token verification fails before converting to HTTP errors."""


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _validate_jwt_config() -> None:
    """
    Validate that JWT verification configuration is present.
    """
    if not settings.wiki_creds:
        raise JWTVerificationError("Missing wiki_creds in configuration.")
    if not settings.jwt_algo:
        raise JWTVerificationError("Missing JWT_ALGO in configuration.")


def _get_verification_secret(token: str) -> str:
    """
    Extract wiki_id from token (unverified) and look up the corresponding secret.
    Falls back to the legacy global secret if no specific wiki secret is found.
    """
    try:
        # Decode without verification just to get the claims
        unverified_payload = jwt.decode(token, options={"verify_signature": False})
        wiki_id = unverified_payload.get("wiki_id")
        
        if wiki_id and wiki_id in settings.wiki_creds:
            return settings.wiki_creds[wiki_id].mw_to_mcp_secret.get_secret_value()
            
    except jwt.PyJWTError:
        pass  # If we can't decode, validation will fail later anyway

    raise JWTVerificationError("No verification secret available for this token. 'wiki_id' matches no configured credentials.")


def _decode_mw_token(token: str) -> dict:
    """
    Decode and validate a JWT issued by MWAssistant.

    Returns
    -------
    dict
        The decoded payload.

    Raises
    ------
    Various JWT-related exceptions, which the public wrapper handles.
    """
    _validate_jwt_config()
    
    secret = _get_verification_secret(token)

    return jwt.decode(
        token,
        secret,
        algorithms=[settings.jwt_algo],
        audience="mw-mcp-server",
        issuer="MWAssistant",
        options={
            "require": ["iss", "aud", "iat", "exp", "user", "user_id", "roles", "scope", "wiki_id"],
        },
    )


_API_URL_WARNED: Set[str] = set()


def _is_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    return parsed._replace(
        scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower()
    ).geturl().rstrip("/")


def _resolve_api_url(wiki_id: str, token_api_url: object) -> Optional[str]:
    """
    Decide which MediaWiki API URL to call for this tenant.

    The token's ``api_url`` claim selects where the server sends outbound,
    JWT-signed requests, so it must not be blindly trusted (SSRF):

    - If ``api_url`` is configured for the wiki in WIKI_CREDS, it is always
      used, and a token naming a different URL is rejected.
    - Otherwise the token value is used for backward compatibility (with a
      one-time warning per wiki), but only if it is an http(s) URL.
    """
    if token_api_url is not None and not isinstance(token_api_url, str):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="'api_url' claim must be a string.",
        )
    token_url = token_api_url.strip() if token_api_url else None

    creds = settings.wiki_creds.get(wiki_id)
    configured = getattr(creds, "api_url", None) if creds else None

    if configured:
        if token_url and _normalize_url(token_url) != _normalize_url(configured):
            logger.warning(
                "Rejecting token for wiki '%s': api_url claim does not match the configured api_url",
                wiki_id,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token 'api_url' does not match the configured wiki API URL.",
            )
        return configured

    if not token_url:
        return None

    if not _is_http_url(token_url):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 'api_url' must be an absolute http(s) URL.",
        )

    if wiki_id not in _API_URL_WARNED:
        _API_URL_WARNED.add(wiki_id)
        logger.warning(
            "No api_url configured for wiki '%s' in WIKI_CREDS; using the api_url "
            "from the incoming token. Configure api_url to pin the wiki endpoint.",
            wiki_id,
        )
    return token_url


# ---------------------------------------------------------------------
# Public Authentication Dependency
# ---------------------------------------------------------------------

def verify_mw_to_mcp_jwt(
    creds: HTTPAuthorizationCredentials = Depends(security),
) -> UserContext:
    """
    Verify JWT from MWAssistant and construct a UserContext.

    Expected claims:
      - iss: "MWAssistant"
      - aud: "mw-mcp-server"
      - user: MediaWiki username
      - roles: list of MW user groups
      - scope: list of granted operations

    Returns
    -------
    UserContext

    Raises
    ------
    HTTPException(401) for invalid or expired tokens.
    """
    token = creds.credentials

    # -------------------------------------------------------------
    # Decode Token
    # -------------------------------------------------------------
    try:
        # We now require 'user_id' in the token for session persistence
        payload = _decode_mw_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired.",
        )
    except jwt.InvalidAudienceError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token audience.",
        )
    except jwt.InvalidIssuerError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token issuer.",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or malformed token.",
        )
    except JWTVerificationError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT verification configuration error.",
        )

    # ---------------------------------------------------------------
    # Validate required fields
    # ---------------------------------------------------------------
    username = payload.get("user")
    user_id = payload.get("user_id")
    wiki_id = payload.get("wiki_id")
    roles = payload.get("roles")
    scopes = payload.get("scope")
    client_id = payload.get("client_id", "MWAssistant")
    allowed_namespaces = payload.get("allowed_namespaces", [])

    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing 'user' claim.",
        )

    if not user_id:
        # Fallback for old tokens or anon users? (Currently strictly required)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing 'user_id' claim.",
        )

    if not wiki_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing 'wiki_id' claim (required for multi-tenant isolation).",
        )

    if not isinstance(roles, list):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="'roles' claim must be a list.",
        )

    if not isinstance(scopes, list):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="'scope' claim must be a list.",
        )

    # Validate allowed_namespaces is a list of integers
    if not isinstance(allowed_namespaces, list):
        allowed_namespaces = []
    allowed_namespaces = [ns for ns in allowed_namespaces if isinstance(ns, int)]

    api_url = _resolve_api_url(wiki_id, payload.get("api_url"))

    return UserContext(
        username=username,
        user_id=user_id,
        wiki_id=wiki_id,
        roles=roles,
        scopes=scopes,
        client_id=client_id,
        allowed_namespaces=allowed_namespaces,
        api_url=api_url,
    )


# ---------------------------------------------------------------------
# Scope enforcement helper
# ---------------------------------------------------------------------

def require_scopes(*required_scopes: str) -> Callable:
    """
    Create a FastAPI dependency that enforces scope-based access control.

    Example:
        @router.post("/chat")
        async def chat(user = Depends(require_scopes("chat_completion"))):
            ...

    Parameters
    ----------
    *required_scopes : str
        The scopes the user must have.

    Returns
    -------
    Callable
        A dependency function that returns UserContext if allowed.
    """

    def check_scopes(
        user: UserContext = Depends(verify_mw_to_mcp_jwt),
    ) -> UserContext:

        missing = [s for s in required_scopes if s not in user.scopes]

        if missing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required scope(s): {', '.join(missing)}",
            )

        return user

    return check_scopes
