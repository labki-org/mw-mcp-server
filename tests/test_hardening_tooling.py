"""Tests for API-docs gating, request-ID validation and admin dashboard hardening."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from mw_mcp_server import main as main_module
from mw_mcp_server.core.middleware import resolve_request_id
from mw_mcp_server.db import get_async_session


def _client(app) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ---------------------------------------------------------------------
# API docs gating
# ---------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
async def test_api_docs_disabled_by_default(monkeypatch, path):
    monkeypatch.delenv("ENABLE_API_DOCS", raising=False)
    app = main_module.create_app()
    async with _client(app) as client:
        resp = await client.get(path)
    assert resp.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
async def test_api_docs_enabled_via_env(monkeypatch, path):
    monkeypatch.setenv("ENABLE_API_DOCS", "true")
    app = main_module.create_app()
    async with _client(app) as client:
        resp = await client.get(path)
    assert resp.status_code == 200


# ---------------------------------------------------------------------
# Request ID validation
# ---------------------------------------------------------------------

@pytest.mark.parametrize("value", ["abc123", "req-1.2_3", "A" * 64])
def test_valid_request_ids_are_kept(value):
    assert resolve_request_id(value) == value


@pytest.mark.parametrize(
    "value",
    [None, "", "A" * 65, "bad id", "x\r\nInjected: 1", "evil|req=forged", "ü"],
)
def test_invalid_request_ids_are_replaced(value):
    out = resolve_request_id(value)
    assert out != value
    assert len(out) == 16 and all(c in "0123456789abcdef" for c in out)


@pytest.mark.asyncio
async def test_middleware_replaces_malformed_request_id():
    async with _client(main_module.app) as client:
        good = await client.get("/health", headers={"X-Request-ID": "trace-42"})
        bad = await client.get("/health", headers={"X-Request-ID": "a b <script>"})
    assert good.headers["x-request-id"] == "trace-42"
    assert bad.headers["x-request-id"] != "a b <script>"
    assert len(bad.headers["x-request-id"]) == 16


# ---------------------------------------------------------------------
# Admin stats / dashboard
# ---------------------------------------------------------------------

@pytest.fixture
async def stats_client():
    session = AsyncMock()
    result = MagicMock()
    result.__iter__.return_value = iter([])
    session.execute.return_value = result

    async def _get_db():
        yield session

    main_module.app.dependency_overrides[get_async_session] = _get_db
    try:
        async with _client(main_module.app) as client:
            yield client
    finally:
        main_module.app.dependency_overrides = {}


@pytest.mark.asyncio
async def test_admin_key_not_accepted_in_query(stats_client):
    with patch(
        "mw_mcp_server.config.settings.admin_api_key",
        MagicMock(get_secret_value=lambda: "secret"),
    ):
        via_query = await stats_client.get("/stats/usage", params={"key": "secret"})
        via_header = await stats_client.get("/stats/usage", headers={"x-admin-key": "secret"})
    assert via_query.status_code == 403
    assert via_header.status_code == 200


@pytest.mark.asyncio
async def test_admin_key_non_ascii_is_rejected_not_500(stats_client):
    with patch(
        "mw_mcp_server.config.settings.admin_api_key",
        MagicMock(get_secret_value=lambda: "secret"),
    ):
        resp = await stats_client.get(
            "/stats/usage", headers={"x-admin-key": "s\u00e9cret".encode("utf-8")}
        )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_dashboard_has_csp_and_pinned_assets(stats_client):
    resp = await stats_client.get("/stats/dashboard")
    assert resp.status_code == 200
    csp = resp.headers["content-security-policy"]
    assert "default-src 'none'" in csp
    assert "frame-ancestors 'none'" in csp
    nonce = csp.split("'nonce-", 1)[1].split("'", 1)[0]
    assert f'nonce="{nonce}"' in resp.text
    assert resp.headers["cache-control"] == "no-store"

    body = resp.text
    assert "cdn.tailwindcss.com" not in body
    assert "chart.js@4.5.1/" in body
    assert 'integrity="sha384-' in body
    assert "localStorage." not in body
    assert "?key=" not in body and "get('key')" not in body
    # Inline event-handler attributes would be blocked by the CSP.
    assert "onclick=" not in body and "onchange=" not in body


@pytest.mark.asyncio
async def test_dashboard_nonce_is_per_request(stats_client):
    r1 = await stats_client.get("/stats/dashboard")
    r2 = await stats_client.get("/stats/dashboard")
    assert r1.headers["content-security-policy"] != r2.headers["content-security-policy"]
