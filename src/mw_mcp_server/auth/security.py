"""JWT verification for incoming requests from MWAssistant."""
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Callable
from ..config import settings
from .models import UserContext

security = HTTPBearer()


def verify_mw_to_mcp_jwt(
    creds: HTTPAuthorizationCredentials = Depends(security),
) -> UserContext:
    """
    Verify JWT from MWAssistant extension.
    
    Expected JWT claims:
    - iss: "MWAssistant"
    - aud: "mw-mcp-server"
    - iat: issued at timestamp
    - exp: expiration timestamp (iat + 30s)
    - user: MediaWiki username
    - roles: List of user groups
    - scope: List of allowed operations
    
    Raises:
        HTTPException(401): Invalid, expired, or malformed token
        HTTPException(403): Valid token but insufficient scope
    """
    token = creds.credentials
    
    try:
        payload = jwt.decode(
            token,
            settings.jwt_mw_to_mcp_secret,
            algorithms=[settings.JWT_ALGO],
            audience="mw-mcp-server",
            issuer="MWAssistant",
            options={
                "require": ["iss", "aud", "iat", "exp", "user", "roles", "scope"]
            }
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidAudienceError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token audience"
        )
    except jwt.InvalidIssuerError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token issuer"
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}"
        )

    
    # Extract and validate required fields
    username = payload.get("user")
    roles = payload.get("roles", [])
    scopes = payload.get("scope", [])
    client_id = payload.get("client_id", "MWAssistant")
    
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing 'user' claim"
        )
    
    if not isinstance(roles, list):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 'roles' must be a list"
        )
    
    if not isinstance(scopes, list):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 'scope' must be a list"
        )
    
    return UserContext(
        username=username,
        roles=roles,
        scopes=scopes,
        client_id=client_id,
    )


def require_scopes(*required_scopes: str) -> Callable:
    """
    Create a dependency that requires specific scopes.
    
    Usage:
        @router.post("/chat/", dependencies=[Depends(require_scopes("chat_completion"))])
        async def chat(user: UserContext = Depends(verify_mw_to_mcp_jwt)):
            ...
    
    Args:
        *required_scopes: One or more scope strings that must be present
    
    Returns:
        Dependency function that validates scopes
    
    Raises:
        HTTPException(403): User lacks required scope
    """
    def check_scopes(user: UserContext = Depends(verify_mw_to_mcp_jwt)) -> UserContext:
        missing_scopes = [s for s in required_scopes if s not in user.scopes]
        
        if missing_scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required scope(s): {', '.join(missing_scopes)}"
            )
        
        return user
    
    return check_scopes

