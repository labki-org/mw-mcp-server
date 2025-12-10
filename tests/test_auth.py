import pytest
import jwt
import time
from fastapi import HTTPException
from mw_mcp_server.auth.security import verify_mw_to_mcp_jwt, require_scopes
from mw_mcp_server.auth.models import UserContext
from mw_mcp_server.config import settings
from mw_mcp_server.auth.jwt_utils import create_mcp_to_mw_jwt

# Mock settings for testing
settings.jwt_mw_to_mcp_secret = "test-secret-mw-to-mcp-must-be-long-enough"
settings.jwt_mcp_to_mw_secret = "test-secret-mcp-to-mw-must-be-long-enough"
settings.JWT_ALGO = "HS256"

def create_valid_token(
    issuer="MWAssistant",
    audience="mw-mcp-server",
    user="TestUser",
    roles=None,
    scopes=None,
    expired=False,
    secret=None
):
    if roles is None:
        roles = ["user"]
    if scopes is None:
        scopes = ["chat_completion"]
    if secret is None:
        secret = settings.jwt_mw_to_mcp_secret
        
    now = int(time.time())
    iat = now - 3600 if expired else now
    exp = iat - 10 if expired else now + 30
    
    payload = {
        "iss": issuer,
        "aud": audience,
        "iat": iat,
        "exp": exp,
        "user": user,
        "roles": roles,
        "scope": scopes,
        "client_id": "MWAssistant"
    }
    
    return jwt.encode(payload, secret, algorithm="HS256")

class MockCredentials:
    def __init__(self, token):
        self.credentials = token

def test_valid_jwt_accepted():
    token = create_valid_token()
    creds = MockCredentials(token)
    user_context = verify_mw_to_mcp_jwt(creds)
    
    assert user_context.username == "TestUser"
    assert "user" in user_context.roles
    assert "chat_completion" in user_context.scopes

def test_expired_jwt_rejected():
    token = create_valid_token(expired=True)
    creds = MockCredentials(token)
    
    with pytest.raises(HTTPException) as excinfo:
        verify_mw_to_mcp_jwt(creds)
    assert excinfo.value.status_code == 401
    assert "expired" in excinfo.value.detail

def test_wrong_issuer_rejected():
    token = create_valid_token(issuer="WrongIssuer")
    creds = MockCredentials(token)
    
    with pytest.raises(HTTPException) as excinfo:
        verify_mw_to_mcp_jwt(creds)
    assert excinfo.value.status_code == 401
    assert "issuer" in excinfo.value.detail

def test_wrong_audience_rejected():
    token = create_valid_token(audience="wrong-audience")
    creds = MockCredentials(token)
    
    with pytest.raises(HTTPException) as excinfo:
        verify_mw_to_mcp_jwt(creds)
    assert excinfo.value.status_code == 401
    assert "audience" in excinfo.value.detail

def test_missing_scope_rejected():
    # Token has "chat_completion" but we require "search"
    token = create_valid_token(scopes=["chat_completion"])
    creds = MockCredentials(token)
    user = verify_mw_to_mcp_jwt(creds)
    
    # Verify strict dependency check
    check_search = require_scopes("search")
    
    with pytest.raises(HTTPException) as excinfo:
        check_search(user)
    assert excinfo.value.status_code == 403
    assert "Missing required scope" in excinfo.value.detail

def test_wrong_signature_rejected():
    token = create_valid_token(secret="wrong-secret-key-that-is-long-enough")
    creds = MockCredentials(token)
    
    with pytest.raises(HTTPException) as excinfo:
        verify_mw_to_mcp_jwt(creds)
    assert excinfo.value.status_code == 401
    assert "Invalid token" in excinfo.value.detail

def test_mcp_to_mw_jwt_generation():
    token = create_mcp_to_mw_jwt(scopes=["page_read"])
    
    # Verify the generated token
    payload = jwt.decode(
        token,
        settings.jwt_mcp_to_mw_secret,
        algorithms=["HS256"],
        audience="MWAssistant"
    )
    
    assert payload["iss"] == "mw-mcp-server"
    assert payload["aud"] == "MWAssistant"
    assert "page_read" in payload["scope"]
    assert payload["exp"] > payload["iat"]
