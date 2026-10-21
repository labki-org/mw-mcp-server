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
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Optional, Protocol, Tuple
from uuid import UUID

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..auth.models import UserContext
from ..auth.security import require_scopes
from ..config import settings
from ..db import AsyncSessionLocal, ChatMessage, ChatSession, VectorStore
from ..db.rate_limiter import RateLimiter, Reservation
from ..embeddings.embedder import Embedder
from ..llm.client import LLMClient, LLMContextLengthError
from ..llm.context import fit_messages_to_budget, truncate_tool_content
from ..prompts import CHAT_SYSTEM_PROMPT, EDITOR_SYSTEM_PROMPT
from ..tools.base import dispatch_tool_call
from ..tools.definitions import TOOL_DEFINITIONS
from ..tools.search_tools import validate_page_access
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
# Bound the cache so a long-running process with many distinct
# (wiki_id, allowed_namespaces) combos can't grow it unboundedly. When full,
# the oldest entry is evicted; ordering is by insertion order via dict.
# Sized so total resident schema text stays well under 10 MB even with
# schema_cap at its upper bound (~2000 elements × ~30 chars × 2 kinds).
_SCHEMA_CACHE_MAX_ENTRIES = 512
# (wiki_id, namespaces) -> (cached_at, latest_ts, categories, properties)
_schema_titles_cache: Dict[Tuple[str, Optional[Tuple[int, ...]]], Tuple[float, Any, List[str], List[str]]] = {}
# (wiki_id, user_id, namespaces) -> (cached_at, rendered). Per user because the
# listed titles are filtered by that user's page-level read access.
_schema_cache: Dict[Tuple[str, int, Optional[Tuple[int, ...]]], Tuple[float, str]] = {}
# Titles per mwassistant-check-access call (MediaWiki's multi-value limit).
_ACCESS_CHECK_BATCH = 50

# Session list page size bounds.
_SESSION_LIST_MAX_LIMIT = 100

# Shown to the user when the model's output hit the length limit.
_TRUNCATION_NOTE = "\n\n_(This response was cut off because it reached the length limit.)_"
_CONTEXT_TOO_LONG_MESSAGE = (
    "This conversation is too long for the model's context window. "
    "Please start a new chat."
)


def _setting(name: str, default: Any) -> Any:
    """Optional tuning knob that may not be declared on Settings yet."""
    return getattr(settings, name, default)


def _context_budget_tokens() -> int:
    # Budget for history + current turn (system prompt and output excluded).
    return int(_setting("chat_context_token_budget", 60_000))


def _tool_result_max_chars() -> int:
    return int(_setting("tool_result_max_chars", 24_000))


def _token_reservation() -> int:
    # Provisional per-request charge; reconciled to actual usage afterwards.
    return int(_setting("chat_token_reservation", 8_000))


@dataclass
class _TurnUsage:
    """Token usage accumulated across every LLM call of one chat turn."""
    prompt: int = 0
    completion: int = 0

    def add(self, usage: Any) -> None:
        self.prompt += usage.prompt_tokens
        self.completion += usage.completion_tokens

    @property
    def total(self) -> int:
        return self.prompt + self.completion


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
    """Append a tool output message in the format expected by the LLM.

    The serialized result is capped (``tool_result_max_chars``) so a single
    large page can't blow the model's context window.
    """
    loop_messages.append(
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": truncate_tool_content(
                json.dumps(tool_output, default=str), _tool_result_max_chars()
            ),
        }
    )


def _summarize_args(args: Any) -> str:
    """Describe tool arguments for logs without their (user-derived) values."""
    if isinstance(args, dict):
        return "{" + ", ".join(
            f"{k}:{type(v).__name__}[{len(v)}]" if isinstance(v, (str, list, dict))
            else f"{k}:{type(v).__name__}"
            for k, v in args.items()
        ) + "}"
    return f"<{type(args).__name__}>"


def _llm_tool_kwargs(tools_mode: str, iteration: int) -> Dict[str, Any]:
    """Map ChatRequest.tools_mode to LLM call kwargs for a loop iteration.

    - ``auto``: tools offered, model decides.
    - ``none``: no tools offered.
    - ``forced``: the first call must use a tool; later calls are ``auto`` so
      the loop can still terminate with an answer.
    """
    if tools_mode == "none":
        return {"tools": None}
    if tools_mode == "forced" and iteration == 0:
        return {"tools": TOOL_DEFINITIONS, "tool_choice": "required"}
    return {"tools": TOOL_DEFINITIONS}


def _with_truncation_note(content: str, finish_reason: Optional[str]) -> str:
    if finish_reason == "length":
        return (content or "") + _TRUNCATION_NOTE
    return content


async def _reserve_or_429(rate_limiter: RateLimiter, user: UserContext) -> Reservation:
    """Atomically admit the request against the daily budget, or raise 429."""
    usage_status, reservation = await rate_limiter.reserve(
        user.wiki_id, user.user_id, _token_reservation()
    )
    if reservation is None:
        reset_time = usage_status.reset_time.strftime("%Y-%m-%d %H:%M UTC")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Daily token limit exceeded. You have used {usage_status.tokens_used:,} "
                f"of {usage_status.limit:,} tokens today. "
                f"Your limit will reset at {reset_time}."
            ),
        )
    return reservation


async def _reconcile_usage(
    rate_limiter: RateLimiter, reservation: Reservation, usage: _TurnUsage
) -> None:
    """Replace the reservation with actual usage; survives client disconnects."""
    with anyio.CancelScope(shield=True):
        try:
            await rate_limiter.reconcile(reservation, usage.prompt, usage.completion)
        except Exception:
            logger.exception("Failed to reconcile token usage for user %s", reservation.user_id)


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


def _cache_put(cache: Dict[Any, Any], key: Any, value: Any) -> None:
    if key not in cache and len(cache) >= _SCHEMA_CACHE_MAX_ENTRIES:
        # Evict the oldest entry; dicts preserve insertion order in 3.7+.
        cache.pop(next(iter(cache)), None)
    cache[key] = value


async def _filter_readable_titles(titles: List[str], user: UserContext) -> List[str]:
    """Keep only titles the user may read (page-level restrictions included).

    Fails closed: titles in a batch whose access check errors are omitted.
    """
    if not titles:
        return []
    batches = [
        titles[i:i + _ACCESS_CHECK_BATCH] for i in range(0, len(titles), _ACCESS_CHECK_BATCH)
    ]
    results = await asyncio.gather(
        *(validate_page_access(batch, user) for batch in batches), return_exceptions=True
    )
    readable: List[str] = []
    for batch, result in zip(batches, results):
        if isinstance(result, BaseException):
            logger.warning(
                "Schema access check failed (%s); omitting %d titles",
                type(result).__name__, len(batch),
            )
            continue
        readable.extend(t for t in batch if result.get(t))
    return readable


async def _get_schema_context(
    vector_store: VectorStore,
    user: UserContext,
) -> str:
    """Build schema context from vector store for system prompt.

    Only includes categories/properties the user can read: namespace access
    from the JWT, then a per-title check against the wiki (ControlAccess-style
    page restrictions). Titles are cached per (wiki, namespaces) and the
    rendered, access-filtered block per user, each for
    ``_SCHEMA_CACHE_TTL_SECONDS``.
    """
    wiki_id = user.wiki_id
    allowed_namespaces = user.allowed_namespaces
    ns_key = tuple(sorted(allowed_namespaces)) if allowed_namespaces else None
    user_key = (wiki_id, user.user_id, ns_key)
    now = time.monotonic()

    cached = _schema_cache.get(user_key)
    if cached and now - cached[0] < _SCHEMA_CACHE_TTL_SECONDS:
        return cached[1]

    titles_key = (wiki_id, ns_key)
    titles_cached = _schema_titles_cache.get(titles_key)
    if titles_cached and now - titles_cached[0] < _SCHEMA_CACHE_TTL_SECONDS:
        _, latest_ts, cats, props = titles_cached
    else:
        fetch_cats = allowed_namespaces is None or NS_CATEGORY in allowed_namespaces
        fetch_props = allowed_namespaces is None or NS_PROPERTY in allowed_namespaces

        # Sequential on purpose: one AsyncSession must not run concurrent queries.
        latest_ts = await vector_store.get_embedding_last_modified(wiki_id)
        cats = (
            await vector_store.get_pages_by_namespace(wiki_id, NS_CATEGORY) if fetch_cats else []
        )
        props = (
            await vector_store.get_pages_by_namespace(wiki_id, NS_PROPERTY) if fetch_props else []
        )
        _cache_put(_schema_titles_cache, titles_key, (now, latest_ts, cats, props))

    cap = settings.schema_cap
    readable = set(await _filter_readable_titles(cats[:cap] + props[:cap], user))
    shown_cats = [t for t in cats[:cap] if t in readable]
    shown_props = [t for t in props[:cap] if t in readable]

    parts = [f"\n\n[KNOWN SCHEMA ELEMENTS (truncated to first {cap} per kind)]\n"]
    if latest_ts:
        parts.append(f"Index last updated: {latest_ts.strftime('%Y-%m-%d %H:%M UTC')}\n")
    parts.append(f"Categories (~{len(shown_cats)}): " + ", ".join(shown_cats))
    if len(cats) > cap:
        parts.append("...")
    parts.append("\n")
    parts.append(f"Properties (~{len(shown_props)}): " + ", ".join(shown_props))
    if len(props) > cap:
        parts.append("...")
    parts.append("\n[END SCHEMA CONTEXT]\n\n")

    rendered = "".join(parts)
    _cache_put(_schema_cache, user_key, (now, rendered))
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
    reservation = await _reserve_or_429(rate_limiter, user)
    usage = _TurnUsage()
    try:
        return await _chat_turn(req, user, llm, vector_store, embedder, session, usage)
    finally:
        await _reconcile_usage(rate_limiter, reservation, usage)


async def _chat_turn(
    req: ChatRequest,
    user: UserContext,
    llm: LLMClient,
    vector_store: VectorStore,
    embedder: Embedder,
    session: AsyncSession,
    usage: _TurnUsage,
) -> ChatResponse:
    """Body of ``POST /chat/`` after the request has been admitted."""
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
    schema_context = await _get_schema_context(vector_store, user)
    system_prompt = base_prompt + schema_context

    try:
        final_answer, used_tools_log = await _run_tool_loop(
            llm=llm,
            user=user,
            vector_store=vector_store,
            embedder=embedder,
            system_prompt=system_prompt,
            initial_messages=full_context,
            history_len=len(history_llm),
            usage=usage,
            tools_mode=req.tools_mode,
        )
    except LLMContextLengthError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=_CONTEXT_TOO_LONG_MESSAGE,
        ) from exc

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
                    "prompt": usage.prompt,
                    "completion": usage.completion,
                    "total": usage.total,
                },
            },
        )
    )

    if not db_session.title and req.messages:
        first_msg = req.messages[0].content
        db_session.title = first_msg[:100] + ("..." if len(first_msg) > 100 else "")
    # Sessions are listed and expired by updated_at; adding messages doesn't
    # touch the session row by itself.
    db_session.updated_at = func.now()

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
    history_len: int = 0,
    usage: Optional[_TurnUsage] = None,
    tools_mode: str = "auto",
) -> tuple[str, List[Dict[str, Any]]]:
    """Run the LLM tool loop and return (final_answer, tool_log).

    Token usage is accumulated into *usage* as each call completes, so it is
    accurate even if the loop fails part-way.
    """
    usage = usage if usage is not None else _TurnUsage()
    max_loops = settings.max_tool_loops
    budget = _context_budget_tokens()
    loop_messages: List[Dict[str, Any]] = list(initial_messages)
    turn_start = history_len
    used_tools_log: List[Dict[str, Any]] = []
    final_answer: Optional[str] = None

    for loop_count in range(max_loops):
        turn_start = fit_messages_to_budget(loop_messages, budget, turn_start, llm.model)
        try:
            chat_result = await llm.chat(
                system_prompt, loop_messages, **_llm_tool_kwargs(tools_mode, loop_count)
            )
        except LLMContextLengthError:
            raise
        except Exception as exc:
            logger.exception("LLM call failed at iteration %d", loop_count)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LLM provider request failed.",
            ) from exc

        response_msg = chat_result.message
        usage.add(chat_result.usage)

        loop_messages.append(response_msg)
        tool_calls = response_msg.get("tool_calls") or []
        if not tool_calls:
            final_answer = _with_truncation_note(
                response_msg.get("content") or "", chat_result.finish_reason
            )
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
                logger.warning(
                    "Tool %s sent invalid JSON args (%d chars)", func_name, len(func_args_str)
                )
                tool_output: Any = {"error": "Invalid JSON arguments for tool call."}
                tool_log_entry["result"] = tool_output
                _append_tool_result(loop_messages, call_id, tool_output)
                continue

            if not isinstance(parsed_args, dict):
                tool_output = {"error": "Tool arguments must be a JSON object."}
                tool_log_entry["result"] = tool_output
                _append_tool_result(loop_messages, call_id, tool_output)
                continue

            logger.info("Tool call %s args=%s", func_name, _summarize_args(parsed_args))
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
            fit_messages_to_budget(loop_messages, budget, turn_start, llm.model)
            chat_result = await llm.chat(system_prompt, loop_messages, tools=None)
            usage.add(chat_result.usage)
            final_answer = _with_truncation_note(
                chat_result.message.get("content") or "", chat_result.finish_reason
            )
        except Exception:
            logger.exception("Wrap-up LLM call failed after max tool loops")
            final_answer = (
                "I wasn't able to finish researching this — the assistant ran out of tool "
                "iterations and the wrap-up call to the language model failed. "
                "Please try again or rephrase your question."
            )

    return final_answer or "", used_tools_log


# ---------------------------------------------------------------------
# Streaming Chat Route (SSE)
# ---------------------------------------------------------------------

# Cap how much of a tool result we ship back to the UI. Full result still goes
# back to the LLM in the next iteration.
_TOOL_PREVIEW_MAX_BYTES = 4096


def _sse(event_type: str, payload: Dict[str, Any]) -> bytes:
    """Encode a payload as an SSE frame: ``event: <type>\\ndata: <json>\\n\\n``."""
    body = json.dumps(payload, default=str)
    return f"event: {event_type}\ndata: {body}\n\n".encode("utf-8")


def _truncate_for_preview(value: Any) -> Any:
    """Shrink a tool result to a UI-friendly preview, keeping JSON shape."""
    serialized = json.dumps(value, default=str)
    if len(serialized) <= _TOOL_PREVIEW_MAX_BYTES:
        return value
    return {
        "_truncated": True,
        "preview": serialized[:_TOOL_PREVIEW_MAX_BYTES],
    }


async def _persist_streamed_turn(
    session_id: UUID,
    req: ChatRequest,
    final_answer: str,
    used_tools_log: List[Dict[str, Any]],
    usage: _TurnUsage,
    session_factory: Any = None,
) -> None:
    """Save a streamed turn's messages in a dedicated, shielded DB session."""
    session_factory = session_factory or AsyncSessionLocal
    timeout = float(_setting("chat_persist_timeout_seconds", 15.0))
    with anyio.move_on_after(timeout, shield=True) as scope:
        try:
            async with session_factory() as db:
                for m in req.messages:
                    db.add(ChatMessage(session_id=session_id, sender=m.role, content=m.content))
                if final_answer:
                    db.add(
                        ChatMessage(
                            session_id=session_id,
                            sender="assistant",
                            content=final_answer,
                            metadata_={
                                "tools_used": [t["name"] for t in used_tools_log] or None,
                                "tokens": {
                                    "prompt": usage.prompt,
                                    "completion": usage.completion,
                                    "total": usage.total,
                                },
                                "streamed": True,
                            },
                        )
                    )
                values: Dict[str, Any] = {"updated_at": func.now()}
                if req.messages:
                    first = req.messages[0].content
                    title = first[:100] + ("..." if len(first) > 100 else "")
                    values["title"] = func.coalesce(ChatSession.title, title)
                await db.execute(
                    update(ChatSession)
                    .where(ChatSession.session_id == session_id)
                    .values(**values)
                )
                await db.commit()
        except Exception:
            logger.exception("Failed to persist streamed chat session")
    if scope.cancelled_caught:
        logger.error("Timed out persisting streamed chat session %s", session_id)


@router.post(
    "/stream",
    summary="Streaming chat with incremental tool-step events (SSE)",
)
async def chat_stream(
    req: ChatRequest,
    request: Request,
    user: Annotated[UserContext, Depends(require_scopes("chat_completion"))],
    llm: Annotated[LLMClient, Depends(get_llm_client)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    rate_limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
) -> StreamingResponse:
    """Stream chat events over SSE so the UI can render each step as it happens.

    Event schema (each frame is ``event: <type>\\ndata: <json>\\n\\n``):

    - ``session`` — first event; payload ``{"session_id", "created"}``.
    - ``tool_start`` — payload ``{"call_id", "name", "args", "iteration"}``.
    - ``tool_result`` — payload ``{"call_id", "name", "ok", "result_preview", "elapsed_ms"}``.
    - ``assistant_message`` — payload ``{"content", "iteration", "is_final"}``
      (emitted once per LLM iteration; the last one with ``is_final=true`` is the
      user-facing answer).
    - ``error`` — payload ``{"code", "message"}``; stream then closes.
    - ``done`` — payload ``{"session_id", "used_tools", "tokens", "final_content"}``.

    Auth: same as ``POST /chat/`` — JWT in ``Authorization`` header. The
    extension must use ``fetch`` + ``ReadableStream`` (not ``EventSource``)
    to send the header.
    """
    reservation = await _reserve_or_429(rate_limiter, user)
    usage = _TurnUsage()
    try:
        # Resolve session up front so the first event carries an authoritative ID.
        db_session: Optional[ChatSession] = None
        history_llm: List[Dict[str, str]] = []
        created = False

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
            created = True

        full_context = history_llm + _to_llm_messages(req.messages)
        base_prompt = EDITOR_SYSTEM_PROMPT if req.context == "editor" else CHAT_SYSTEM_PROMPT
        schema_context = await _get_schema_context(vector_store, user)
        system_prompt = base_prompt + schema_context

        # Commit the session row now: the turn is persisted from a separate DB
        # session at the end of the stream (see _persist_streamed_turn), which
        # must be able to see it.
        await session.commit()
    except BaseException:
        await _reconcile_usage(rate_limiter, reservation, usage)
        raise

    session_uuid = db_session.session_id
    session_id_str = str(session_uuid)
    history_len = len(history_llm)

    async def event_generator():
        loop_messages: List[Dict[str, Any]] = list(full_context)
        turn_start = history_len
        budget = _context_budget_tokens()
        used_tools_log: List[Dict[str, Any]] = []
        final_answer: str = ""
        max_loops = settings.max_tool_loops

        try:
            yield _sse("session", {"session_id": session_id_str, "created": created})

            for loop_count in range(max_loops):
                if await request.is_disconnected():
                    logger.info("Client disconnected before LLM iteration %d", loop_count)
                    return

                turn_start = fit_messages_to_budget(
                    loop_messages, budget, turn_start, llm.model
                )
                try:
                    chat_result = await llm.chat(
                        system_prompt,
                        loop_messages,
                        **_llm_tool_kwargs(req.tools_mode, loop_count),
                    )
                except LLMContextLengthError:
                    logger.warning("Chat context too long at iteration %d", loop_count)
                    yield _sse(
                        "error",
                        {"code": "context_length", "message": _CONTEXT_TOO_LONG_MESSAGE},
                    )
                    return
                except Exception:
                    logger.exception("LLM call failed at iteration %d", loop_count)
                    yield _sse(
                        "error",
                        {"code": "llm_failure", "message": "LLM provider request failed."},
                    )
                    return

                response_msg = chat_result.message
                usage.add(chat_result.usage)
                loop_messages.append(response_msg)

                tool_calls = response_msg.get("tool_calls") or []
                content = response_msg.get("content") or ""
                is_final = not tool_calls
                if is_final:
                    content = _with_truncation_note(content, chat_result.finish_reason)

                if content or is_final:
                    yield _sse(
                        "assistant_message",
                        {
                            "content": content,
                            "iteration": loop_count,
                            "is_final": is_final,
                        },
                    )

                if is_final:
                    final_answer = content
                    break

                for tc in tool_calls:
                    if await request.is_disconnected():
                        logger.info("Client disconnected mid-tool")
                        return

                    func_name = tc["function"]["name"]
                    func_args_str = tc["function"]["arguments"]
                    call_id = tc["id"]

                    tool_log_entry: Dict[str, Any] = {
                        "name": func_name,
                        "args": func_args_str,
                    }
                    used_tools_log.append(tool_log_entry)

                    try:
                        parsed_args = json.loads(func_args_str)
                    except json.JSONDecodeError:
                        parsed_args = None

                    if not isinstance(parsed_args, dict):
                        msg = (
                            "Invalid JSON arguments for tool call."
                            if parsed_args is None
                            else "Tool arguments must be a JSON object."
                        )
                        tool_output: Any = {"error": msg}
                        tool_log_entry["result"] = tool_output
                        _append_tool_result(loop_messages, call_id, tool_output)
                        yield _sse(
                            "tool_start",
                            {
                                "call_id": call_id,
                                "name": func_name,
                                "args": {},
                                "iteration": loop_count,
                            },
                        )
                        yield _sse(
                            "tool_result",
                            {
                                "call_id": call_id,
                                "name": func_name,
                                "ok": False,
                                "result_preview": tool_output,
                                "elapsed_ms": 0,
                            },
                        )
                        continue

                    yield _sse(
                        "tool_start",
                        {
                            "call_id": call_id,
                            "name": func_name,
                            "args": parsed_args,
                            "iteration": loop_count,
                        },
                    )
                    logger.info("Tool call %s args=%s", func_name, _summarize_args(parsed_args))
                    started = time.monotonic()
                    ok = True
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
                        ok = False

                    elapsed_ms = int((time.monotonic() - started) * 1000)
                    tool_log_entry["result"] = tool_output
                    _append_tool_result(loop_messages, call_id, tool_output)
                    yield _sse(
                        "tool_result",
                        {
                            "call_id": call_id,
                            "name": func_name,
                            "ok": ok,
                            "result_preview": _truncate_for_preview(tool_output),
                            "elapsed_ms": elapsed_ms,
                        },
                    )
            else:
                # Loop exhausted — force a tool-free wrap-up call.
                try:
                    fit_messages_to_budget(loop_messages, budget, turn_start, llm.model)
                    chat_result = await llm.chat(system_prompt, loop_messages, tools=None)
                    usage.add(chat_result.usage)
                    final_answer = _with_truncation_note(
                        chat_result.message.get("content") or "", chat_result.finish_reason
                    )
                except Exception:
                    logger.exception("Wrap-up LLM call failed after max tool loops")
                    final_answer = (
                        "I wasn't able to finish researching this — the assistant ran out "
                        "of tool iterations and the wrap-up call to the language model "
                        "failed. Please try again or rephrase your question."
                    )
                yield _sse(
                    "assistant_message",
                    {
                        "content": final_answer,
                        "iteration": max_loops,
                        "is_final": True,
                    },
                )
        finally:
            # Always persist + record usage, even on disconnect or error. On a
            # client disconnect this runs while the task is being cancelled
            # (or from the async-generator finalizer), so it is shielded and
            # uses its own DB session: the request-scoped session may already
            # be torn down, and an unshielded await here would be cancelled
            # immediately, letting the turn skip the daily token limit.
            await _reconcile_usage(rate_limiter, reservation, usage)
            await _persist_streamed_turn(
                session_uuid, req, final_answer, used_tools_log, usage
            )

        yield _sse(
            "done",
            {
                "session_id": session_id_str,
                "used_tools": used_tools_log,
                "tokens": {
                    "prompt": usage.prompt,
                    "completion": usage.completion,
                    "total": usage.total,
                },
                "final_content": final_answer,
            },
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            # Disable proxy buffering so events flush immediately.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


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
    """List chat sessions for the authenticated user.

    ``limit`` is clamped to 1..100 and ``offset`` to >= 0.
    """
    limit = max(1, min(limit, _SESSION_LIST_MAX_LIMIT))
    offset = max(0, offset)
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
