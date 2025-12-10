"""JWT utility functions for generating and verifying tokens."""
import jwt
import time
from typing import List
from ..config import settings


def create_mcp_to_mw_jwt(scopes: List[str]) -> str:
    """
    Generate a short-lived JWT for MCP → MW communication.
    
    Args:
        scopes: List of scopes this token grants (e.g., ["page_read", "smw_query"])
    
    Returns:
        Encoded JWT string
    """
    now = int(time.time())
    payload = {
        "iss": "mw-mcp-server",
        "aud": "MWAssistant",
        "iat": now,
        "exp": now + settings.JWT_TTL,
        "scope": scopes,
    }
    
    return jwt.encode(
        payload,
        settings.jwt_mcp_to_mw_secret,
        algorithm=settings.JWT_ALGO
    )
