import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, ASGITransport
from datetime import date, timedelta
from typing import List, Any

from mw_mcp_server.main import app
from mw_mcp_server.db import get_async_session

# Helper to create mock DB rows
class MockRow:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

@pytest.fixture
def mock_db_session():
    session = AsyncMock()
    return session

@pytest.fixture
def override_get_db(mock_db_session):
    async def _get_db():
        yield mock_db_session
    app.dependency_overrides[get_async_session] = _get_db
    yield
    app.dependency_overrides = {}

@pytest.fixture
async def async_client(override_get_db):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

@pytest.mark.asyncio
async def test_admin_dashboard_security(async_client):
    """Verify admin endpoints are secured."""
    
    # 1. No key -> 403
    resp = await async_client.get("/stats/usage")
    assert resp.status_code == 403
    
    # 2. Wrong key -> 403
    resp = await async_client.get("/stats/usage", params={"key": "wrong"})
    assert resp.status_code == 403
    
    # 3. For success tests, we need to mock the setting.
    # We'll use a patch context manager in the test.

@pytest.mark.asyncio
async def test_usage_aggregation_logic(async_client, mock_db_session):
    """Verify stats aggregation deals with DB results correctly."""
    
    # Setup mocks for 4 queries: Token, Session, Message, Embedding
    
    # 1. Token Usage Result
    # wiki_id, total_tokens, prompt_tokens, completion_tokens, request_count, active_users
    token_rows = [
        MockRow(
            wiki_id="wiki-1", 
            total_tokens=1000, 
            prompt_tokens=800, 
            completion_tokens=200, 
            request_count=10, 
            active_users=5
        ),
        MockRow(
            wiki_id="wiki-2", 
            total_tokens=500, 
            prompt_tokens=400, 
            completion_tokens=100, 
            request_count=2, 
            active_users=1
        )
    ]
    
    # 2. Session Count Result
    # wiki_id, session_count
    session_rows = [
        MockRow(wiki_id="wiki-1", session_count=3),
        MockRow(wiki_id="wiki-2", session_count=1)
    ]
    
    # 3. Message Count Result
    # wiki_id, message_count
    message_rows = [
        MockRow(wiki_id="wiki-1", message_count=15),
        MockRow(wiki_id="wiki-2", message_count=5)
    ]
    
    # 4. Embedding Count Result
    # wiki_id, embedding_count
    embedding_rows = [
        MockRow(wiki_id="wiki-1", embedding_count=50),
        MockRow(wiki_id="wiki-2", embedding_count=0) # maybe missing row
    ]

    # Configure db.execute to return these in order
    # Each execute result needs to be iterable
    
    # We can use side_effect to return different mocks for successive calls
    async def execute_side_effect(query):
        # We can try to inspect the query or just return strict order if we know it
        # The code executes: token, session, message, embedding
        return MagicMock() # Placeholder

    # To make result iterable, we need the return value of execute() to support __iter__ 
    # But db.execute is async, so it returns a Result object that we iterate or fetchall.
    # The code does: for row in await db.execute(...)
    
    # So execute return value should be an iterable of rows
    mock_db_session.execute.side_effect = [
        token_rows,
        session_rows,
        message_rows,
        embedding_rows
    ]

    # Patch settings to allow access
    with patch("mw_mcp_server.config.settings.admin_api_key", MagicMock(get_secret_value=lambda: "secret")):
        resp = await async_client.get("/stats/usage", headers={"x-admin-key": "secret"})
        
    assert resp.status_code == 200
    data = resp.json()
    
    assert len(data["tenants"]) == 2
    
    # Verify Wiki 1
    w1 = next(t for t in data["tenants"] if t["wiki_id"] == "wiki-1")
    assert w1["total_tokens"] == 1000
    assert w1["session_count"] == 3
    assert w1["message_count"] == 15
    assert w1["embedding_count"] == 50
    assert w1["active_users"] == 5
    
    # Verify Wiki 2
    w2 = next(t for t in data["tenants"] if t["wiki_id"] == "wiki-2")
    assert w2["total_tokens"] == 500
    assert w2["active_users"] == 1
    assert w2["embedding_count"] == 0 # Defaulted to 0 due to missing row or just explicit 0

@pytest.mark.asyncio
async def test_dashboard_html(async_client):
    """Verify dashboard HTML is served."""
    resp = await async_client.get("/stats/dashboard")
    assert resp.status_code == 200
    assert "Admin Dashboard" in resp.text
