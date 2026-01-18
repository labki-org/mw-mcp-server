
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
import jwt
import time
import contextlib

from mw_mcp_server.main import app
from mw_mcp_server.api.dependencies import get_vector_store, get_embedder
from mw_mcp_server.db import VectorStore
from mw_mcp_server.embeddings.embedder import Embedder

# Test secrets
TEST_MW_TO_MCP_SECRET = "test-secret-mw-to-mcp-must-be-long-enough-32chars"
TEST_JWT_ALGO = "HS256"

def create_valid_token(
    user="TestUser",
    wiki_id="test-wiki",
    roles=None,
    scopes=None,
    secret=TEST_MW_TO_MCP_SECRET
):
    if roles is None:
        roles = ["user"]
    if scopes is None:
        scopes = ["embeddings", "chat_completion"]
        
    now = int(time.time())
    payload = {
        "iss": "MWAssistant",
        "aud": "mw-mcp-server",
        "iat": now,
        "exp": now + 3600,
        "user": user,
        "user_id": 123,
        "wiki_id": wiki_id,
        "roles": roles,
        "scope": scopes,
        "client_id": "MWAssistant"
    }
    return jwt.encode(payload, secret, algorithm=TEST_JWT_ALGO)

@pytest.fixture
def mock_vectors():
    mock = AsyncMock(spec=VectorStore)
    mock.get_stats.return_value = {
        "total_vectors": 100, 
        "total_pages": 10,
        "page_timestamps": {}
    }
    mock.delete_page.return_value = 5
    mock.add_documents.return_value = 10
    return mock

@pytest.fixture
def mock_embedder():
    mock = AsyncMock(spec=Embedder)
    # Mock return of embed: list of vectors (list of floats)
    # 2 chunks -> 2 vectors
    mock.embed.return_value = [[0.1]*1536, [0.2]*1536]
    return mock

@pytest.fixture
def client(mock_vectors, mock_embedder):
    app.dependency_overrides[get_vector_store] = lambda: mock_vectors
    app.dependency_overrides[get_embedder] = lambda: mock_embedder
    
    # Mock lifespan to avoid DB connection
    @contextlib.asynccontextmanager
    async def mock_lifespan(app):
        yield
        
    app.router.lifespan_context = mock_lifespan
    
    with TestClient(app) as c:
        yield c
        
    app.dependency_overrides = {}

@pytest.fixture
def mock_settings():
    with patch("mw_mcp_server.auth.security.settings") as mock:
        from mw_mcp_server.config import WikiCredentials
        from pydantic import SecretStr
        
        mock.wiki_creds = {
            "test-wiki": WikiCredentials(
                mw_to_mcp_secret=SecretStr(TEST_MW_TO_MCP_SECRET),
                mcp_to_mw_secret=SecretStr("irrelevant")
            )
        }
        mock.jwt_algo = TEST_JWT_ALGO
        yield mock

def test_get_stats(client, mock_settings, mock_vectors):
    token = create_valid_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    resp = client.get("/embeddings/stats", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["total_vectors"] == 100
    mock_vectors.get_stats.assert_awaited_once()

def test_update_page_embedding(client, mock_settings, mock_vectors, mock_embedder):
    token = create_valid_token()
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "title": "Test Page",
        "content": "This is a test paragraph.\n\nThis is another paragraph.",
        "namespace": 0,
        "last_modified": "2023-01-01T12:00:00Z"
    }
    
    # Mock the queue
    with patch("mw_mcp_server.embeddings.queue.embedding_queue.enqueue", new_callable=AsyncMock) as mock_enqueue:
        mock_enqueue.return_value = 1  # Queue size

        resp = client.post("/embeddings/page", json=payload, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        
        # Verify enqueue called
        mock_enqueue.assert_awaited_once()
        
        # Verify background work did NOT happen synchronously
        mock_vectors.delete_page.assert_not_called()
        mock_embedder.embed.assert_not_called()
        mock_vectors.add_documents.assert_not_called()
        mock_vectors.commit.assert_not_called()

def test_delete_page_embedding(client, mock_settings, mock_vectors):
    token = create_valid_token()
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "title": "Test Page To Delete"
    }
    
    # Method is DELETE, but payload is body?
    # Standard DELETEs don't always have bodies, but FastApi supports it.
    # However, client.delete(..., json=...) works in TestClient.
    
    resp = client.request("DELETE", "/embeddings/page", json=payload, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "deleted"
    
    mock_vectors.delete_page.assert_awaited_once()
    mock_vectors.commit.assert_awaited_once()

def test_missing_scope_rejected(client, mock_settings):
    # Token without "embeddings" scope
    token = create_valid_token(scopes=["chat_completion"])
    headers = {"Authorization": f"Bearer {token}"}
    
    resp = client.get("/embeddings/stats", headers=headers)
    assert resp.status_code == 403
