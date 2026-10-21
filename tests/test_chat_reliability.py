"""
Chat route reliability tests: rate-limit reservation, context budgeting,
tool-result caps, tools_mode, truncation notes, session list clamping and
streamed-turn persistence on client disconnect.

Route functions are called directly with fakes; no database or network.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest
from fastapi import HTTPException

from mw_mcp_server.api import chat_routes
from mw_mcp_server.api.models import ChatMessage as ChatMessageModel
from mw_mcp_server.api.models import ChatRequest
from mw_mcp_server.auth.models import UserContext
from mw_mcp_server.db.rate_limiter import Reservation, UsageStatus
from mw_mcp_server.llm.client import ChatResult, LLMTransportError, TokenUsage


def _user():
    return UserContext(
        username="TestUser",
        wiki_id="test-wiki",
        user_id=1,
        client_id="test",
        allowed_namespaces=[0, 14, 102],
    )


def _req(**kw):
    return ChatRequest(messages=[ChatMessageModel(role="user", content="hello")], **kw)


def _result(content="answer", tool_calls=None, finish_reason="stop"):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return ChatResult(
        message=msg,
        usage=TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        finish_reason=finish_reason,
    )


def _tool_call(name="mw_get_page", args='{"title": "Big"}', call_id="c1"):
    return [{"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}]


class FakeLLM:
    model = "gpt-4.1-mini"

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def chat(self, system_prompt, messages, tools=None, **kw):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools, **kw})
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _limiter(admitted=True):
    limiter = MagicMock()
    reservation = Reservation("test-wiki", 1, datetime.now(timezone.utc).date(), 8000)
    status = UsageStatus(
        tokens_used=0 if admitted else 999_999,
        tokens_remaining=1,
        limit=100_000,
        requests_today=1,
        is_limited=not admitted,
        reset_time=datetime.now(timezone.utc),
    )
    limiter.reserve = AsyncMock(return_value=(status, reservation if admitted else None))
    limiter.reconcile = AsyncMock()
    return limiter


def _vector_store():
    vs = MagicMock()
    vs.get_embedding_last_modified = AsyncMock(return_value=None)
    vs.get_pages_by_namespace = AsyncMock(return_value=[])
    return vs


def _db():
    db = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.execute = AsyncMock()
    return db


@pytest.fixture(autouse=True)
def _clear_schema_cache():
    chat_routes._schema_cache.clear()
    chat_routes._schema_titles_cache.clear()
    yield
    chat_routes._schema_cache.clear()
    chat_routes._schema_titles_cache.clear()


async def _chat(req, llm, limiter=None, db=None):
    return await chat_routes.chat(
        req=req,
        user=_user(),
        llm=llm,
        vector_store=_vector_store(),
        embedder=MagicMock(),
        session=db or _db(),
        rate_limiter=limiter or _limiter(),
    )


# ---------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------

async def test_over_limit_returns_429_without_calling_llm():
    llm = FakeLLM([])
    with pytest.raises(HTTPException) as exc:
        await _chat(_req(), llm, limiter=_limiter(admitted=False))
    assert exc.value.status_code == 429
    assert llm.calls == []


async def test_usage_reconciled_with_actual_tokens():
    limiter = _limiter()
    await _chat(_req(), FakeLLM([_result()]), limiter=limiter)
    reservation, prompt, completion = limiter.reconcile.await_args.args
    assert (prompt, completion) == (100, 20)
    assert reservation.reserved_tokens == 8000


async def test_usage_reconciled_even_when_llm_fails():
    limiter = _limiter()
    llm = FakeLLM([_result(content=None, tool_calls=_tool_call()), LLMTransportError("x")])
    with patch.object(chat_routes, "dispatch_tool_call", AsyncMock(return_value={"ok": 1})):
        with pytest.raises(HTTPException) as exc:
            await _chat(_req(), llm, limiter=limiter)
    assert exc.value.status_code == 502
    # The first (successful) call's tokens are still charged.
    assert limiter.reconcile.await_args.args[1:] == (100, 20)


# ---------------------------------------------------------------------
# Context growth
# ---------------------------------------------------------------------

async def test_tool_results_are_capped_before_reaching_llm(monkeypatch):
    monkeypatch.setattr(chat_routes, "_tool_result_max_chars", lambda: 500)
    llm = FakeLLM([_result(content=None, tool_calls=_tool_call()), _result()])
    huge = {"content": "x" * 50_000}
    with patch.object(chat_routes, "dispatch_tool_call", AsyncMock(return_value=huge)):
        await _chat(_req(), llm)

    tool_msg = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"][0]
    assert len(tool_msg["content"]) < 700
    assert "truncated" in tool_msg["content"]


async def test_long_history_is_trimmed_to_budget(monkeypatch):
    monkeypatch.setattr(chat_routes, "_context_budget_tokens", lambda: 2_000)
    history = []
    for i in range(30):
        history.append(MagicMock(role=None, sender="user", content=f"question {i} " * 100))
        history.append(MagicMock(role=None, sender="assistant", content=f"answer {i} " * 100))
    db_session = MagicMock(messages=history, session_id="sid", title="t")
    llm = FakeLLM([_result()])

    with patch.object(chat_routes, "_load_user_session", AsyncMock(return_value=db_session)):
        await _chat(_req(session_id="00000000-0000-0000-0000-000000000000"), llm)

    sent = llm.calls[0]["messages"]
    assert len(sent) < 61
    assert sent[-1]["content"] == "hello"  # current turn always kept
    assert sent[0]["role"] == "user"
    assert "question 29" in sent[-3]["content"]  # newest history kept, oldest dropped


async def test_context_length_error_returns_413():
    from mw_mcp_server.llm.client import LLMContextLengthError

    llm = FakeLLM([LLMContextLengthError("too long")])
    with pytest.raises(HTTPException) as exc:
        await _chat(_req(), llm)
    assert exc.value.status_code == 413


async def test_finish_reason_length_appends_note():
    llm = FakeLLM([_result(content="partial", finish_reason="length")])
    resp = await _chat(_req(), llm)
    assert resp.messages[-1].content.startswith("partial")
    assert "cut off" in resp.messages[-1].content


async def test_tools_mode_none_and_forced():
    llm = FakeLLM([_result()])
    await _chat(_req(tools_mode="none"), llm)
    assert llm.calls[0]["tools"] is None

    llm = FakeLLM([_result(content=None, tool_calls=_tool_call()), _result()])
    with patch.object(chat_routes, "dispatch_tool_call", AsyncMock(return_value={})):
        await _chat(_req(tools_mode="forced"), llm)
    assert llm.calls[0]["tool_choice"] == "required"
    assert "tool_choice" not in llm.calls[1]


def test_summarize_args_hides_values():
    summary = chat_routes._summarize_args({"title": "Secret Page", "limit": 5})
    assert "Secret" not in summary
    assert summary == "{title:str[11], limit:int}"


# ---------------------------------------------------------------------
# Schema context
# ---------------------------------------------------------------------

async def test_schema_context_filters_unreadable_titles():
    vs = _vector_store()
    vs.get_pages_by_namespace = AsyncMock(
        side_effect=lambda wiki, ns: ["Category:Open", "Category:Secret"] if ns == 14 else []
    )
    access = AsyncMock(return_value={"Category:Open": True, "Category:Secret": False})
    with patch.object(chat_routes, "validate_page_access", access):
        rendered = await chat_routes._get_schema_context(vs, _user())
    assert "Category:Open" in rendered
    assert "Category:Secret" not in rendered


async def test_schema_context_fails_closed_on_access_error():
    vs = _vector_store()
    vs.get_pages_by_namespace = AsyncMock(return_value=["Category:A"])
    with patch.object(
        chat_routes, "validate_page_access", AsyncMock(side_effect=RuntimeError("down"))
    ):
        rendered = await chat_routes._get_schema_context(vs, _user())
    assert "Category:A" not in rendered


# ---------------------------------------------------------------------
# Session list
# ---------------------------------------------------------------------

@pytest.mark.parametrize("limit,expected", [(10_000, 100), (0, 1), (-5, 1), (20, 20)])
async def test_session_list_limit_is_clamped(limit, expected):
    db = _db()
    db.execute = AsyncMock(return_value=MagicMock(all=lambda: []))
    await chat_routes.list_sessions(user=_user(), session=db, limit=limit, offset=-3)
    stmt = db.execute.await_args.args[0]
    compiled = stmt.compile(compile_kwargs={"literal_binds": True})
    assert f"LIMIT {expected}" in str(compiled)
    assert "OFFSET 0" in str(compiled)


# ---------------------------------------------------------------------
# Streaming persistence on disconnect
# ---------------------------------------------------------------------

class _RecordingSession:
    """Async-context-manager DB session whose awaits yield to the loop."""

    def __init__(self, log):
        self.log = log

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def add(self, obj):
        self.log.append(("add", obj.sender, obj.content))

    async def execute(self, stmt):
        await anyio.sleep(0.01)
        self.log.append(("execute",))

    async def commit(self):
        await anyio.sleep(0.01)
        self.log.append(("commit",))


async def test_stream_persists_and_reconciles_when_client_disconnects(monkeypatch):
    persisted = []
    monkeypatch.setattr(chat_routes, "AsyncSessionLocal", lambda: _RecordingSession(persisted))

    llm_started = anyio.Event()

    class BlockingLLM(FakeLLM):
        async def chat(self, system_prompt, messages, tools=None, **kw):
            if self.results:
                return await super().chat(system_prompt, messages, tools, **kw)
            llm_started.set()
            await anyio.sleep_forever()

    llm = BlockingLLM([_result(content=None, tool_calls=_tool_call())])
    limiter = _limiter()

    async def slow_reconcile(*args):
        await anyio.sleep(0.01)  # would be cancelled without the shield

    limiter.reconcile = AsyncMock(side_effect=slow_reconcile)
    request = MagicMock()
    request.is_disconnected = AsyncMock(return_value=False)

    with patch.object(chat_routes, "dispatch_tool_call", AsyncMock(return_value={"ok": 1})):
        response = await chat_routes.chat_stream(
            req=_req(),
            request=request,
            user=_user(),
            llm=llm,
            vector_store=_vector_store(),
            embedder=MagicMock(),
            session=_db(),
            rate_limiter=limiter,
        )

        async def consume():
            async for _ in response.body_iterator:
                pass

        # Starlette (ASGI spec < 2.4) cancels the streaming task through an
        # anyio cancel scope when the client disconnects; emulate that.
        async with anyio.create_task_group() as tg:
            tg.start_soon(consume)
            await llm_started.wait()
            tg.cancel_scope.cancel()

    limiter.reconcile.assert_awaited_once()
    assert limiter.reconcile.await_args.args[1:] == (100, 20)
    assert ("add", "user", "hello") in persisted
    assert ("commit",) in persisted


def test_sse_frames_are_json():
    frame = chat_routes._sse("done", {"a": 1}).decode()
    assert frame.startswith("event: done\n")
    assert json.loads(frame.split("data: ", 1)[1]) == {"a": 1}
