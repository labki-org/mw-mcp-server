"""
Chat Routes: LLM-Driven Conversational Interface

This module implements the core conversational endpoint used by the MediaWiki
extension. It provides:
- Multi-turn chat session support backed by PostgreSQL
- Tool invocation loops based on LLM function calls
- Secure, scope-controlled access with full user context
- Structured output suitable for the MediaWiki client

Major Responsibilities
----------------------
1. Accept ChatRequest containing user messages and optional session_id.
2. Retrieve prior session history from database (if applicable).
3. Call the LLM with:
    - System prompt
    - Combined history + new user messages
    - Tool definitions
4. Detect and execute LLM tool/function calls.
5. Make subsequent LLM calls with the tool results.
6. Save final messages to the database.
7. Return ChatResponse with final assistant message + tool usage log.

Security Model
--------------
- The route requires the `chat_completion` scope.
- Authentication via JWT is handled upstream by the `require_scopes` dependency.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Annotated, Any, Awaitable, Dict, List, Optional, Protocol, Tuple
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..auth.models import UserContext
from ..auth.security import require_scopes
from ..config import settings
from ..db import ChatMessage, ChatSession, VectorStore
from ..db.rate_limiter import RateLimiter
from ..embeddings.embedder import Embedder
from ..llm.client import LLMClient
from ..prompts import CHAT_SYSTEM_PROMPT, EDITOR_SYSTEM_PROMPT
from ..tools.base import dispatch_tool_call
from ..tools.definitions import TOOL_DEFINITIONS
from .dependencies import (
    get_db_session,
    get_embedder,
    get_llm_client,
    get_rate_limiter,
    get_vector_store,
)
from .models import ChatMessage as ChatMessageModel
from .models import ChatRequest, ChatResponse

logger = logging.getLogger("mcp.chat")

router = APIRouter(prefix="/chat", tags=["chat"])

# MediaWiki namespace IDs
NS_CATEGORY = 14
NS_PROPERTY = 102

# In-process TTL cache for schema context. Schema only changes when embeddings
# are added/deleted (via the background worker). 60s of staleness is acceptable
# and saves 2-3 DB queries per chat turn.
_SCHEMA_CACHE_TTL_SECONDS = 60.0
_schema_cache: Dict[Tuple[str, Optional[Tuple[int, ...]]], Tuple[float, str]] = {}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


class _HasRoleAndContent(Protocol):
    role: str
    content: str


def _to_llm_messages(messages: List[Any]) -> List[Dict[str, str]]:
    """Convert request or DB message objects into the OpenAI message dict shape.

    Accepts any object exposing ``.role`` (e.g. ChatMessageModel) or ``.sender``
    (DB ChatMessage) plus ``.content``.
    """
    return [
        {
            "role": getattr(m, "role", None) or m.sender,
            "content": m.content,
        }
        for m in messages
    ]


def _append_tool_result(
    loop_messages: List[Dict[str, Any]],
    call_id: str,
    tool_output: Any,
) -> None:
    """Append a tool output message in the format expected by the LLM."""
    loop_messages.append(
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(tool_output, default=str),
        }
    )


async def _load_user_session(
    session: AsyncSession,
    session_id: str,
    user: UserContext,
    *,
    with_messages: bool = False,
) -> Optional[ChatSession]:
    """Return the user-owned session matching ``session_id`` or None.

    Returns None for missing sessions and for malformed UUIDs.
    """
    try:
        session_uuid = UUID(session_id)
    except (ValueError, TypeError):
        return None

    stmt = (
        select(ChatSession)
        .where(ChatSession.session_id == session_uuid)
        .where(ChatSession.wiki_id == user.wiki_id)
        .where(ChatSession.owner_user_id == user.user_id)
    )
    if with_messages:
        stmt = stmt.options(selectinload(ChatSession.messages))

    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def _get_schema_context(
    vector_store: VectorStore,
    wiki_id: str,
    allowed_namespaces: Optional[list] = None,
) -> str:
    """Build schema context from vector store for system prompt.

    Only includes categories/properties the user has namespace access to.
    Cached in-process for ``_SCHEMA_CACHE_TTL_SECONDS``.
    """
    cache_key = (wiki_id, tuple(sorted(allowed_namespaces)) if allowed_namespaces else None)
    cached = _schema_cache.get(cache_key)
    now = time.monotonic()
    if cached and now - cached[0] < _SCHEMA_CACHE_TTL_SECONDS:
        return cached[1]

    fetch_cats = allowed_namespaces is None or NS_CATEGORY in allowed_namespaces
    fetch_props = allowed_namespaces is None or NS_PROPERTY in allowed_namespaces

    awaitables: List[Awaitable[Any]] = [vector_store.get_embedding_last_modified(wiki_id)]
    if fetch_cats:
        awaitables.append(vector_store.get_pages_by_namespace(wiki_id, NS_CATEGORY))
    if fetch_props:
        awaitables.append(vector_store.get_pages_by_namespace(wiki_id, NS_PROPERTY))

    results = await asyncio.gather(*awaitables)
    latest_ts = results[0]
    idx = 1
    cats: List[str] = results[idx] if fetch_cats else []
    if fetch_cats:
        idx += 1
    props: List[str] = results[idx] if fetch_props else []

    cap = settings.schema_cap
    parts = ["\n\n[KNOWN SCHEMA ELEMENTS (Truncated if > 100)]\n"]
    if latest_ts:
        parts.append(f"Index last updated: {latest_ts.strftime('%Y-%m-%d %H:%M UTC')}\n")
    parts.append(f"Categories (~{len(cats)}): " + ", ".join(cats[:cap]))
    if len(cats) > cap:
        parts.append("...")
    parts.append("\n")
    parts.append(f"Properties (~{len(props)}): " + ", ".join(props[:cap]))
    if len(props) > cap:
        parts.append("...")
    parts.append("\n[END SCHEMA CONTEXT]\n\n")

    rendered = "".join(parts)
    _schema_cache[cache_key] = (now, rendered)
    return rendered


# ---------------------------------------------------------------------
# Chat Route
# ---------------------------------------------------------------------


@router.post(
    "/",
    response_model=ChatResponse,
    summary="Chat with the MediaWiki LLM Assistant",
    status_code=status.HTTP_200_OK,
)
async def chat(
    req: ChatRequest,
    user: Annotated[UserContext, Depends(require_scopes("chat_completion"))],
    llm: Annotated[LLMClient, Depends(get_llm_client)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    rate_limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
) -> ChatResponse:
    """
    Core conversational endpoint for MediaWiki-assisted LLM interactions.

    Parameters
    ----------
    req : ChatRequest
        Contains:
        - session_id: Optional session UUID
        - messages: Current user messages (list of ChatMessage)

    user : UserContext
        Authenticated user identity extracted from JWT.

    Returns
    -------
    ChatResponse
        Final assistant message + which tools were used.
    """
    usage_status = await rate_limiter.check_limit(user.wiki_id, user.user_id)
    if usage_status.is_limited:
        reset_time = usage_status.reset_time.strftime("%Y-%m-%d %H:%M UTC")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Daily token limit exceeded. You have used {usage_status.tokens_used:,} "
                f"of {usage_status.limit:,} tokens today. "
                f"Your limit will reset at {reset_time}."
            ),
        )

    db_session: Optional[ChatSession] = None
    history_llm: List[Dict[str, str]] = []

    if req.session_id:
        db_session = await _load_user_session(
            session, req.session_id, user, with_messages=True
        )
        if db_session:
            history_llm = _to_llm_messages(db_session.messages)

    if not db_session:
        db_session = ChatSession(wiki_id=user.wiki_id, owner_user_id=user.user_id)
        session.add(db_session)
        await session.flush()

    full_context = history_llm + _to_llm_messages(req.messages)

    base_prompt = EDITOR_SYSTEM_PROMPT if req.context == "editor" else CHAT_SYSTEM_PROMPT
    schema_context = await _get_schema_context(
        vector_store, user.wiki_id, allowed_namespaces=user.allowed_namespaces
    )
    system_prompt = base_prompt + schema_context

    final_answer, used_tools_log, total_prompt_tokens, total_completion_tokens = (
        await _run_tool_loop(
            llm=llm,
            user=user,
            vector_store=vector_store,
            embedder=embedder,
            system_prompt=system_prompt,
            initial_messages=full_context,
        )
    )

    await rate_limiter.record_usage(
        wiki_id=user.wiki_id,
        user_id=user.user_id,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
    )

    for msg in req.messages:
        session.add(
            ChatMessage(
                session_id=db_session.session_id,
                sender=msg.role,
                content=msg.content,
            )
        )

    session.add(
        ChatMessage(
            session_id=db_session.session_id,
            sender="assistant",
            content=final_answer,
            metadata_={
                "tools_used": [t["name"] for t in used_tools_log] if used_tools_log else None,
                "tokens": {
                    "prompt": total_prompt_tokens,
                    "completion": total_completion_tokens,
                    "total": total_prompt_tokens + total_completion_tokens,
                },
            },
        )
    )

    if not db_session.title and req.messages:
        first_msg = req.messages[0].content
        db_session.title = first_msg[:100] + ("..." if len(first_msg) > 100 else "")

    return ChatResponse(
        messages=req.messages + [ChatMessageModel(role="assistant", content=final_answer)],
        used_tools=used_tools_log,
        session_id=str(db_session.session_id),
    )


async def _run_tool_loop(
    *,
    llm: LLMClient,
    user: UserContext,
    vector_store: VectorStore,
    embedder: Embedder,
    system_prompt: str,
    initial_messages: List[Dict[str, Any]],
) -> tuple[str, List[Dict[str, Any]], int, int]:
    """Run the LLM tool loop and return (final_answer, tool_log, prompt_tokens, completion_tokens)."""
    max_loops = settings.max_tool_loops
    loop_messages: List[Dict[str, Any]] = list(initial_messages)
    used_tools_log: List[Dict[str, Any]] = []
    prompt_tokens = 0
    completion_tokens = 0
    final_answer: Optional[str] = None

    for loop_count in range(max_loops):
        try:
            chat_result = await llm.chat(system_prompt, loop_messages, tools=TOOL_DEFINITIONS)
        except Exception as exc:
            logger.exception("LLM call failed at iteration %d", loop_count)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LLM provider request failed.",
            ) from exc

        response_msg = chat_result.message
        prompt_tokens += chat_result.usage.prompt_tokens
        completion_tokens += chat_result.usage.completion_tokens

        loop_messages.append(response_msg)
        tool_calls = response_msg.get("tool_calls") or []
        if not tool_calls:
            final_answer = response_msg.get("content") or ""
            break

        for tc in tool_calls:
            func_name = tc["function"]["name"]
            func_args_str = tc["function"]["arguments"]
            call_id = tc["id"]

            tool_log_entry: Dict[str, Any] = {"name": func_name, "args": func_args_str}
            used_tools_log.append(tool_log_entry)

            try:
                parsed_args = json.loads(func_args_str)
            except json.JSONDecodeError:
                logger.warning("Tool %s sent invalid JSON args: %r", func_name, func_args_str)
                tool_output: Any = {"error": "Invalid JSON arguments for tool call."}
                tool_log_entry["result"] = tool_output
                _append_tool_result(loop_messages, call_id, tool_output)
                continue

            if not isinstance(parsed_args, dict):
                tool_output = {"error": "Tool arguments must be a JSON object."}
                tool_log_entry["result"] = tool_output
                _append_tool_result(loop_messages, call_id, tool_output)
                continue

            try:
                tool_output = await dispatch_tool_call(
                    func_name,
                    parsed_args,
                    user,
                    vector_store=vector_store,
                    embedder=embedder,
                )
            except Exception as exc:
                logger.exception("Tool %s execution failed", func_name)
                tool_output = {"error": f"Tool execution failed: {type(exc).__name__}"}

            tool_log_entry["result"] = tool_output
            _append_tool_result(loop_messages, call_id, tool_output)
    else:
        # Loop exhausted without a tool-free assistant turn — force one final LLM call
        # without tools so we get a user-facing answer.
        try:
            chat_result = await llm.chat(system_prompt, loop_messages, tools=None)
            prompt_tokens += chat_result.usage.prompt_tokens
            completion_tokens += chat_result.usage.completion_tokens
            final_answer = chat_result.message.get("content") or ""
        except Exception:
            logger.exception("Wrap-up LLM call failed after max tool loops")
            final_answer = (
                "I wasn't able to finish researching this — the assistant ran out of tool "
                "iterations and the wrap-up call to the language model failed. "
                "Please try again or rephrase your question."
            )

    return final_answer or "", used_tools_log, prompt_tokens, completion_tokens


# ---------------------------------------------------------------------
# Session Management Routes
# ---------------------------------------------------------------------


def _iso(dt: Any) -> Optional[str]:
    return dt.isoformat() if dt else None


@router.get(
    "/sessions",
    summary="List user's chat sessions",
)
async def list_sessions(
    user: Annotated[UserContext, Depends(require_scopes("chat_completion"))],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    limit: int = 50,
    offset: int = 0,
) -> List[dict]:
    """List chat sessions for the authenticated user."""
    message_count = (
        select(func.count(ChatMessage.message_id))
        .where(ChatMessage.session_id == ChatSession.session_id)
        .correlate(ChatSession)
        .scalar_subquery()
    )
    result = await session.execute(
        select(ChatSession, message_count.label("message_count"))
        .where(ChatSession.wiki_id == user.wiki_id)
        .where(ChatSession.owner_user_id == user.user_id)
        .order_by(ChatSession.updated_at.desc())
        .offset(offset)
        .limit(limit)
    )

    return [
        {
            "session_id": str(s.session_id),
            "title": s.title,
            "created_at": _iso(s.created_at),
            "updated_at": _iso(s.updated_at),
            "message_count": count,
        }
        for s, count in result.all()
    ]


@router.get(
    "/sessions/{session_id}",
    summary="Get a specific chat session",
)
async def get_session(
    session_id: str,
    user: Annotated[UserContext, Depends(require_scopes("chat_completion"))],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> dict:
    """Get full conversation history for a session."""
    db_session = await _load_user_session(session, session_id, user, with_messages=True)
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": str(db_session.session_id),
        "title": db_session.title,
        "created_at": _iso(db_session.created_at),
        "updated_at": _iso(db_session.updated_at),
        "messages": [
            {
                "role": m.sender,
                "content": m.content,
                "created_at": _iso(m.created_at),
            }
            for m in db_session.messages
        ],
    }


@router.delete(
    "/sessions/{session_id}",
    summary="Delete a chat session",
)
async def delete_session(
    session_id: str,
    user: Annotated[UserContext, Depends(require_scopes("chat_completion"))],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> dict:
    """Delete a chat session and all its messages."""
    db_session = await _load_user_session(session, session_id, user)
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")

    await session.delete(db_session)
    return {"status": "deleted", "session_id": session_id}
