"""
JWT Authentication Tests

Tests for JWT verification and scope enforcement.
Uses environment variables set by CI workflow for test secrets.
"""

import pytest
import jwt
import time
import os
from unittest.mock import patch
from fastapi import HTTPException
from pydantic import SecretStr

from mw_mcp_server.auth.security import verify_mw_to_mcp_jwt, require_scopes
from mw_mcp_server.auth.jwt_utils import create_mcp_to_mw_jwt


# Test secrets - use environment variables if available (CI), otherwise defaults
TEST_MW_TO_MCP_SECRET = os.getenv(
    "JWT_MW_TO_MCP_SECRET", 
    "test-secret-mw-to-mcp-must-be-long-enough-32chars"
)
TEST_MCP_TO_MW_SECRET = os.getenv(
    "JWT_MCP_TO_MW_SECRET",
    "test-secret-mcp-to-mw-must-be-long-enough-32chars"
)
TEST_JWT_ALGO = "HS256"


def create_valid_token(
    issuer="MWAssistant",
    audience="mw-mcp-server",
    user="TestUser",
    wiki_id="test-wiki",
    roles=None,
    scopes=None,
    expired=False,
    secret=None
):
    """Create a JWT token for testing."""
    if roles is None:
        roles = ["user"]
    if scopes is None:
        scopes = ["chat_completion"]
    if secret is None:
        secret = TEST_MW_TO_MCP_SECRET
        
    now = int(time.time())
    iat = now - 3600 if expired else now
    exp = iat - 10 if expired else now + 30
    
    payload = {
        "iss": issuer,
        "aud": audience,
        "iat": iat,
        "exp": exp,
        "user": user,
        "wiki_id": wiki_id,
        "roles": roles,
        "scope": scopes,
        "client_id": "MWAssistant"
    }
    
    return jwt.encode(payload, secret, algorithm=TEST_JWT_ALGO)


class MockCredentials:
    """Mock HTTPAuthorizationCredentials for testing."""
    def __init__(self, token):
        self.credentials = token


@pytest.fixture
def mock_settings():
    """Fixture that patches settings with test values."""
    with patch("mw_mcp_server.auth.security.settings") as mock:
        mock.jwt_mw_to_mcp_secret = SecretStr(TEST_MW_TO_MCP_SECRET)
        mock.jwt_algo = TEST_JWT_ALGO
        yield mock


@pytest.fixture
def mock_jwt_utils_settings():
    """Fixture that patches settings for jwt_utils."""
    with patch("mw_mcp_server.auth.jwt_utils.settings") as mock:
        mock.jwt_mcp_to_mw_secret = SecretStr(TEST_MCP_TO_MW_SECRET)
        mock.jwt_algo = TEST_JWT_ALGO
        mock.jwt_ttl_seconds = 30
        yield mock


class TestJWTVerification:
    """Tests for JWT token verification."""

    def test_valid_jwt_accepted(self, mock_settings):
        """Valid JWT should be accepted and return UserContext."""
        token = create_valid_token()
        creds = MockCredentials(token)
        user_context = verify_mw_to_mcp_jwt(creds)
        
        assert user_context.username == "TestUser"
        assert user_context.wiki_id == "test-wiki"
        assert "user" in user_context.roles
        assert "chat_completion" in user_context.scopes

    def test_expired_jwt_rejected(self, mock_settings):
        """Expired JWT should be rejected with 401."""
        token = create_valid_token(expired=True)
        creds = MockCredentials(token)
        
        with pytest.raises(HTTPException) as excinfo:
            verify_mw_to_mcp_jwt(creds)
        assert excinfo.value.status_code == 401
        assert "expired" in excinfo.value.detail.lower()

    def test_wrong_issuer_rejected(self, mock_settings):
        """JWT with wrong issuer should be rejected."""
        token = create_valid_token(issuer="WrongIssuer")
        creds = MockCredentials(token)
        
        with pytest.raises(HTTPException) as excinfo:
            verify_mw_to_mcp_jwt(creds)
        assert excinfo.value.status_code == 401
        assert "issuer" in excinfo.value.detail.lower()

    def test_wrong_audience_rejected(self, mock_settings):
        """JWT with wrong audience should be rejected."""
        token = create_valid_token(audience="wrong-audience")
        creds = MockCredentials(token)
        
        with pytest.raises(HTTPException) as excinfo:
            verify_mw_to_mcp_jwt(creds)
        assert excinfo.value.status_code == 401
        assert "audience" in excinfo.value.detail.lower()

    def test_wrong_signature_rejected(self, mock_settings):
        """JWT signed with wrong secret should be rejected."""
        token = create_valid_token(secret="wrong-secret-key-that-is-long-enough")
        creds = MockCredentials(token)
        
        with pytest.raises(HTTPException) as excinfo:
            verify_mw_to_mcp_jwt(creds)
        assert excinfo.value.status_code == 401
        # Could be "Invalid" or "malformed" depending on PyJWT version
        assert "invalid" in excinfo.value.detail.lower() or "malformed" in excinfo.value.detail.lower()


class TestScopeEnforcement:
    """Tests for scope-based access control."""

    def test_missing_scope_rejected(self, mock_settings):
        """Request without required scope should be rejected with 403."""
        token = create_valid_token(scopes=["chat_completion"])
        creds = MockCredentials(token)
        user = verify_mw_to_mcp_jwt(creds)
        
        # Require a scope the user doesn't have
        check_search = require_scopes("search")
        
        with pytest.raises(HTTPException) as excinfo:
            check_search(user)
        assert excinfo.value.status_code == 403
        assert "Missing required scope" in excinfo.value.detail

    def test_has_required_scope_accepted(self, mock_settings):
        """Request with required scope should pass."""
        token = create_valid_token(scopes=["chat_completion", "search"])
        creds = MockCredentials(token)
        user = verify_mw_to_mcp_jwt(creds)
        
        check_search = require_scopes("search")
        result = check_search(user)
        
        assert result.username == "TestUser"
        assert "search" in result.scopes


class TestMCPToMWJWT:
    """Tests for MCP server issuing JWTs to MediaWiki."""

    def test_mcp_to_mw_jwt_generation(self, mock_jwt_utils_settings):
        """MCP server should generate valid JWTs for MediaWiki."""
        token = create_mcp_to_mw_jwt(scopes=["page_read"])
        
        # Verify the generated token
        payload = jwt.decode(
            token,
            TEST_MCP_TO_MW_SECRET,
            algorithms=[TEST_JWT_ALGO],
            audience="MWAssistant"
        )
        
        assert payload["iss"] == "mw-mcp-server"
        assert payload["aud"] == "MWAssistant"
        assert "page_read" in payload["scope"]
        assert payload["exp"] > payload["iat"]
