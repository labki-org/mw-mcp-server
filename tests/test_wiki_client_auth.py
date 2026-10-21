"""
MCP -> MediaWiki Request Tests

Covers the wire contract with MWAssistant:
- signed wiki_id / user / user_id / jti claims on outbound JWTs
- username/user_id params kept in sync with those claims
- mwassistant-check-access batching (max 100 titles per request)
- api_url pinning from WIKI_CREDS (SSRF) and http(s)-only targets
- no redirect following, redacted error logs
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List
from unittest.mock import patch
from urllib.parse import parse_qs

import httpx
import jwt
import pytest
from fastapi import HTTPException
from pydantic import SecretStr

from mw_mcp_server.auth import security
from mw_mcp_server.auth.jwt_utils import JWTConfigurationError, create_mcp_to_mw_jwt
from mw_mcp_server.auth.models import UserContext
from mw_mcp_server.config import WikiCredentials
from mw_mcp_server.wiki.api_client import (
    CHECK_ACCESS_BATCH_SIZE,
    MediaWikiClient,
    MediaWikiRequestError,
)
from mw_mcp_server.wiki.smw_client import SMWClient, SMWQueryError

MCP_TO_MW_SECRET = os.getenv(
    "JWT_MCP_TO_MW_SECRET", "test-secret-mcp-to-mw-must-be-long-enough-32chars"
)
MW_TO_MCP_SECRET = os.getenv(
    "JWT_MW_TO_MCP_SECRET", "test-secret-mw-to-mcp-must-be-long-enough-32chars"
)
API_URL = "https://wiki.example.org/w/api.php"


def _creds(api_url: str | None = None) -> Dict[str, WikiCredentials]:
    return {
        "test-wiki": WikiCredentials(
            mw_to_mcp_secret=SecretStr(MW_TO_MCP_SECRET),
            mcp_to_mw_secret=SecretStr(MCP_TO_MW_SECRET),
            api_url=api_url,
        )
    }


@pytest.fixture
def jwt_settings():
    with patch("mw_mcp_server.auth.jwt_utils.settings") as mock:
        mock.wiki_creds = _creds()
        mock.jwt_algo = "HS256"
        mock.jwt_ttl_seconds = 30
        yield mock


@pytest.fixture
def user() -> UserContext:
    return UserContext(
        username="Alice",
        user_id=42,
        wiki_id="test-wiki",
        client_id="MWAssistant",
        allowed_namespaces=[0],
        api_url=API_URL,
    )


def _decode(token: str) -> Dict[str, Any]:
    return jwt.decode(token, MCP_TO_MW_SECRET, algorithms=["HS256"], audience="MWAssistant")


class Recorder:
    """httpx MockTransport handler that records requests."""

    def __init__(self, responder=None):
        self.requests: List[httpx.Request] = []
        self.responder = responder or (lambda req: {"ok": True})

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = self.responder(request)
        if isinstance(body, httpx.Response):
            return body
        return httpx.Response(200, json=body)

    def params(self, request: httpx.Request) -> Dict[str, str]:
        if request.method == "POST":
            raw = parse_qs(request.content.decode())
        else:
            raw = parse_qs(request.url.query.decode())
        return {k: v[0] for k, v in raw.items()}

    def claims(self, request: httpx.Request) -> Dict[str, Any]:
        return _decode(request.headers["Authorization"].removeprefix("Bearer "))


def _client(recorder: Recorder) -> MediaWikiClient:
    client = MediaWikiClient()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    return client


# ---------------------------------------------------------------------
# create_mcp_to_mw_jwt
# ---------------------------------------------------------------------

class TestMcpToMwClaims:
    def test_system_token_has_wiki_id_and_jti_but_no_user(self, jwt_settings):
        claims = _decode(create_mcp_to_mw_jwt(["page_read"], wiki_id="test-wiki"))
        assert claims["wiki_id"] == "test-wiki"
        assert claims["jti"]
        assert "user" not in claims
        assert "user_id" not in claims

    def test_jti_is_unique(self, jwt_settings):
        a = _decode(create_mcp_to_mw_jwt(["page_read"], wiki_id="test-wiki"))
        b = _decode(create_mcp_to_mw_jwt(["page_read"], wiki_id="test-wiki"))
        assert a["jti"] != b["jti"]

    def test_user_token_carries_identity(self, jwt_settings, user):
        claims = _decode(create_mcp_to_mw_jwt(["page_read"], wiki_id="test-wiki", user=user))
        assert claims["user"] == "Alice"
        assert claims["user_id"] == 42
        assert isinstance(claims["user_id"], int)  # MW requires an int claim

    def test_anonymous_user_sends_only_user_id_zero(self, jwt_settings):
        anon = UserContext.model_construct(
            username="127.0.0.1", user_id=0, wiki_id="test-wiki",
            client_id="x", roles=[], scopes=[], allowed_namespaces=[], api_url=None,
        )
        claims = _decode(create_mcp_to_mw_jwt(["page_read"], wiki_id="test-wiki", user=anon))
        assert claims["user_id"] == 0
        assert "user" not in claims

    def test_refuses_user_of_other_wiki(self, jwt_settings, user):
        jwt_settings.wiki_creds = {**_creds(), "other": _creds()["test-wiki"]}
        with pytest.raises(JWTConfigurationError):
            create_mcp_to_mw_jwt(["page_read"], wiki_id="other", user=user)


# ---------------------------------------------------------------------
# MediaWikiClient requests
# ---------------------------------------------------------------------

class TestClientRequests:
    async def test_page_fetch_binds_user_claims_and_params(self, jwt_settings, user):
        rec = Recorder(lambda req: {"mwassistant-page": {"exists": True, "wikitext": "x"}})
        await _client(rec).get_page_wikitext("Main Page", user=user)

        req = rec.requests[0]
        assert str(req.url).startswith(API_URL)
        params = rec.params(req)
        claims = rec.claims(req)
        assert params["username"] == claims["user"] == "Alice"
        assert int(params["user_id"]) == claims["user_id"] == 42
        assert claims["wiki_id"] == "test-wiki"

    async def test_keyword_search_sends_user_id(self, jwt_settings, user):
        rec = Recorder(lambda req: {"mwassistant-keyword-search": []})
        await _client(rec).search_pages("foo", user=user)
        params = rec.params(rec.requests[0])
        assert params["username"] == "Alice"
        assert params["user_id"] == "42"
        assert rec.claims(rec.requests[0])["user"] == "Alice"

    async def test_smw_ask_unwraps_result_and_binds_user(self, jwt_settings, user):
        payload = {"rows": [], "printouts": [], "count": 0, "has_more": False, "errors": []}
        rec = Recorder(lambda req: {"mwassistant-smw": payload})
        out = await SMWClient(_client(rec)).ask("[[Category:X]]", user=user)
        assert out == payload
        assert rec.claims(rec.requests[0])["user_id"] == 42
        assert rec.params(rec.requests[0])["user_id"] == "42"

    async def test_redirects_are_not_followed(self, jwt_settings, user):
        rec = Recorder(lambda req: httpx.Response(302, headers={"Location": "http://169.254.169.254/"}))
        client = MediaWikiClient()
        # Use the real client construction to verify follow_redirects=False.
        real = await client._get_client()
        assert real.follow_redirects is False
        await client.aclose()

        client = _client(rec)
        with pytest.raises(MediaWikiRequestError):
            await client.get_page_wikitext("Main Page", user=user)
        assert len(rec.requests) == 1

    async def test_non_http_api_url_rejected(self, jwt_settings):
        client = _client(Recorder())
        with pytest.raises(MediaWikiRequestError):
            await client.request({"action": "query"}, api_url="file:///etc/passwd", wiki_id="test-wiki")

    async def test_error_log_redacts_user_params(self, jwt_settings, user, caplog):
        rec = Recorder(lambda req: httpx.Response(500))
        caplog.set_level(logging.ERROR, logger="mcp.mediawiki")
        with pytest.raises(MediaWikiRequestError):
            await _client(rec).get_page_wikitext("Secret Page", user=user)
        text = caplog.text
        assert "Alice" not in text
        assert "Secret Page" not in text
        assert "mwassistant-page" in text

    async def test_smw_error_log_omits_query(self, jwt_settings, user, caplog):
        rec = Recorder(lambda req: httpx.Response(500))
        caplog.set_level(logging.ERROR, logger="mcp.smw")
        with pytest.raises(SMWQueryError):
            await SMWClient(_client(rec)).ask("[[Has secret::Value]]", user=user)
        assert "Has secret" not in caplog.text
        assert "len=" in caplog.text


# ---------------------------------------------------------------------
# check-access batching
# ---------------------------------------------------------------------

class TestCheckAccessBatching:
    async def test_batches_of_at_most_100_titles_via_post(self, jwt_settings, user):
        def respond(req: httpx.Request):
            titles = parse_qs(req.content.decode())["titles"][0].split("|")
            assert len(titles) <= CHECK_ACCESS_BATCH_SIZE
            return {"mwassistant-check-access": {
                "access": {t: not t.endswith("7") for t in titles}
            }}

        rec = Recorder(respond)
        titles = [f"Page {i}" for i in range(250)]
        access = await _client(rec).check_read_access(titles, user)

        assert CHECK_ACCESS_BATCH_SIZE == 100
        assert len(rec.requests) == 3
        assert all(r.method == "POST" for r in rec.requests)
        assert set(access) == set(titles)
        assert access["Page 1"] is True
        assert access["Page 17"] is False
        for req in rec.requests:
            params = rec.params(req)
            assert params["username"] == "Alice"
            assert params["user_id"] == "42"
            assert rec.claims(req)["user"] == "Alice"

    async def test_duplicates_checked_once(self, jwt_settings, user):
        rec = Recorder(lambda req: {"mwassistant-check-access": {
            "access": {t: True for t in parse_qs(req.content.decode())["titles"][0].split("|")}
        }})
        out = await _client(rec).check_read_access(["A", "B", "A"], user)
        assert out == {"A": True, "B": True}
        assert rec.params(rec.requests[0])["titles"] == "A|B"

    async def test_empty_titles_make_no_request(self, jwt_settings, user):
        rec = Recorder()
        assert await _client(rec).check_read_access([], user) == {}
        assert rec.requests == []


# ---------------------------------------------------------------------
# api_url resolution (SSRF)
# ---------------------------------------------------------------------

def _incoming_token(api_url=None) -> str:
    now = int(time.time())
    payload = {
        "iss": "MWAssistant", "aud": "mw-mcp-server", "iat": now, "exp": now + 30,
        "user": "Alice", "user_id": 42, "wiki_id": "test-wiki",
        "roles": [], "scope": ["chat_completion"],
    }
    if api_url is not None:
        payload["api_url"] = api_url
    return jwt.encode(payload, MW_TO_MCP_SECRET, algorithm="HS256")


class _Creds:
    def __init__(self, token):
        self.credentials = token


class TestApiUrlResolution:
    @pytest.fixture
    def sec_settings(self):
        with patch("mw_mcp_server.auth.security.settings") as mock:
            mock.wiki_creds = _creds()
            mock.jwt_algo = "HS256"
            security._API_URL_WARNED.clear()
            yield mock

    def test_configured_url_overrides_missing_claim(self, sec_settings):
        sec_settings.wiki_creds = _creds(API_URL)
        ctx = security.verify_mw_to_mcp_jwt(_Creds(_incoming_token()))
        assert ctx.api_url == API_URL

    def test_matching_claim_accepted(self, sec_settings):
        sec_settings.wiki_creds = _creds(API_URL)
        ctx = security.verify_mw_to_mcp_jwt(_Creds(_incoming_token(API_URL)))
        assert ctx.api_url == API_URL

    def test_mismatching_claim_rejected(self, sec_settings):
        sec_settings.wiki_creds = _creds(API_URL)
        with pytest.raises(HTTPException) as exc:
            security.verify_mw_to_mcp_jwt(_Creds(_incoming_token("http://169.254.169.254/latest")))
        assert exc.value.status_code == 401

    def test_unconfigured_uses_claim_and_warns_once(self, sec_settings, caplog):
        caplog.set_level(logging.WARNING, logger="mcp.auth")
        for _ in range(2):
            ctx = security.verify_mw_to_mcp_jwt(_Creds(_incoming_token(API_URL)))
            assert ctx.api_url == API_URL
        assert caplog.text.count("No api_url configured") == 1

    def test_unconfigured_rejects_non_http_claim(self, sec_settings):
        with pytest.raises(HTTPException) as exc:
            security.verify_mw_to_mcp_jwt(_Creds(_incoming_token("gopher://internal:70/")))
        assert exc.value.status_code == 401

    def test_config_rejects_non_http_api_url(self):
        with pytest.raises(ValueError):
            WikiCredentials(
                mw_to_mcp_secret=SecretStr("x" * 32),
                mcp_to_mw_secret=SecretStr("y" * 32),
                api_url="file:///etc/passwd",
            )

    def test_wiki_creds_json_accepts_api_url(self):
        from mw_mcp_server.config import Settings

        s = Settings(
            openai_api_key="sk-test",
            wiki_creds=json.dumps({"w": {
                "mw_to_mcp_secret": "a" * 32, "mcp_to_mw_secret": "b" * 32,
                "api_url": API_URL,
            }}),
        )
        assert s.wiki_creds["w"].api_url == API_URL
