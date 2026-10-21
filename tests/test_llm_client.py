"""
LLM client and context-budget tests.
"""

import json

import httpx
import pytest
import respx

from mw_mcp_server.llm.client import (
    LLMClient,
    LLMContextLengthError,
    LLMTransportError,
)
from mw_mcp_server.llm.context import (
    ELIDED_TOOL_RESULT,
    fit_messages_to_budget,
    truncate_tool_content,
)

URL = "https://api.openai.com/v1/chat/completions"


def _ok(content="hi", finish_reason="stop"):
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"role": "assistant", "content": content},
                 "finish_reason": finish_reason}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )


def _client(**kw):
    kw.setdefault("max_retries", 3)
    return LLMClient(api_key="sk-test", model="gpt-4.1-mini", retry_base_delay=0, **kw)


# ---------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------

@respx.mock
async def test_retries_429_then_succeeds():
    route = respx.post(URL).mock(
        side_effect=[httpx.Response(429, headers={"retry-after": "0"}), _ok()]
    )
    result = await _client().chat("sys", [{"role": "user", "content": "q"}])
    assert route.call_count == 2
    assert result.message["content"] == "hi"


@respx.mock
async def test_retries_timeouts_and_5xx_then_gives_up():
    route = respx.post(URL).mock(
        side_effect=[httpx.ReadTimeout("t"), httpx.Response(503), httpx.Response(502)]
    )
    with pytest.raises(LLMTransportError):
        await _client(max_retries=2).chat("sys", [{"role": "user", "content": "q"}])
    assert route.call_count == 3  # 1 try + 2 retries, bounded


@respx.mock
async def test_does_not_retry_client_errors():
    route = respx.post(URL).mock(return_value=httpx.Response(401))
    with pytest.raises(LLMTransportError):
        await _client().chat("sys", [{"role": "user", "content": "q"}])
    assert route.call_count == 1


@respx.mock
async def test_context_length_error_is_distinguished():
    respx.post(URL).mock(
        return_value=httpx.Response(
            400, json={"error": {"code": "context_length_exceeded", "message": "too long"}}
        )
    )
    with pytest.raises(LLMContextLengthError):
        await _client().chat("sys", [{"role": "user", "content": "q"}])


@respx.mock
async def test_finish_reason_and_options_in_payload():
    route = respx.post(URL).mock(return_value=_ok(finish_reason="length"))
    result = await _client().chat(
        "sys",
        [{"role": "user", "content": "q"}],
        tools=[{"type": "function", "function": {"name": "t"}}],
        tool_choice="required",
        max_tokens=100,
    )
    assert result.finish_reason == "length"
    body = json.loads(route.calls.last.request.content)
    assert body["tool_choice"] == "required"
    assert body["max_completion_tokens"] == 100


# ---------------------------------------------------------------------
# Context budget
# ---------------------------------------------------------------------

def _msg(role, n_chars, **kw):
    return {"role": role, "content": "x " * (n_chars // 2), **kw}


def test_truncate_tool_content():
    assert truncate_tool_content("short", 100) == "short"
    out = truncate_tool_content("a" * 1000, 100)
    assert out.startswith("a" * 100)
    assert "truncated 900 characters" in out


def test_fit_drops_oldest_history_but_keeps_current_turn():
    history = [_msg("user", 4000), _msg("assistant", 4000)] * 5
    current = [_msg("user", 200)]
    messages = history + current

    turn_start = fit_messages_to_budget(messages, 3000, len(history))

    assert messages[-1] is current[0]
    assert turn_start < len(history)
    assert turn_start == len(messages) - 1
    # Never starts with an orphaned assistant reply.
    assert messages[0]["role"] == "user"


def test_fit_elides_older_tool_results_before_dropping_history():
    history = [_msg("user", 400), _msg("assistant", 400)]
    turn = [
        _msg("user", 100),
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "1", "function": {"name": "get", "arguments": "{}"}}]},
        _msg("tool", 40_000, tool_call_id="1"),
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "2", "function": {"name": "get", "arguments": "{}"}}]},
        _msg("tool", 4_000, tool_call_id="2"),
    ]
    messages = history + turn

    turn_start = fit_messages_to_budget(messages, 4_000, len(history))

    assert turn_start == 2  # history kept: eliding the big tool result was enough
    assert messages[4]["content"] == ELIDED_TOOL_RESULT
    assert messages[4]["tool_call_id"] == "1"  # pairing preserved
    assert messages[6]["content"] != ELIDED_TOOL_RESULT  # latest result kept


def test_fit_is_noop_under_budget():
    messages = [_msg("user", 100)]
    assert fit_messages_to_budget(messages, 10_000, 0) == 0
    assert len(messages) == 1
