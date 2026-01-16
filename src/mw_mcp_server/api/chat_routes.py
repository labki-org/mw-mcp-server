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

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated, List, Dict, Any, Optional
from uuid import UUID
import json

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import ChatRequest, ChatResponse, ChatMessage as ChatMessageModel
from ..auth.models import UserContext
from ..auth.security import require_scopes
from ..llm.client import LLMClient
from ..db import VectorStore, ChatSession, ChatMessage
from ..db.rate_limiter import RateLimiter
from ..embeddings.embedder import Embedder
from ..tools.definitions import TOOL_DEFINITIONS
from ..tools.base import dispatch_tool_call
from .dependencies import get_llm_client, get_embedder, get_db_session, get_vector_store, get_rate_limiter
from ..prompts import CHAT_SYSTEM_PROMPT, EDITOR_SYSTEM_PROMPT

router = APIRouter(prefix="/chat", tags=["chat"])


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def _convert_db_messages_to_llm_format(messages: List[ChatMessage]) -> List[Dict[str, str]]:
    """Convert database ChatMessage objects into plain dicts for LLM input."""
    return [{"role": m.sender, "content": m.content} for m in messages]


def _convert_request_messages_to_llm_format(messages: List[ChatMessageModel]) -> List[Dict[str, str]]:
    """Convert request ChatMessage objects into plain dicts for LLM input."""
    return [{"role": m.role, "content": m.content} for m in messages]


def _append_tool_result(
    loop_messages: List[Dict[str, Any]],
    call_id: str,
    tool_output: Any,
) -> None:
    """Append a tool output message in the format expected by the LLM."""
    loop_messages.append({
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(tool_output, default=str),
    })


async def _get_schema_context(vector_store: VectorStore, wiki_id: str) -> str:
    """Build schema context from vector store for system prompt."""
    SCHEMA_CAP = 100 
    
    # NS_CATEGORY = 14, NS_PROPERTY = 102
    cats = await vector_store.get_pages_by_namespace(wiki_id, 14)
    props = await vector_store.get_pages_by_namespace(wiki_id, 102)
    
    schema_context = "\n\n[KNOWN SCHEMA ELEMENTS (Truncated if > 100)]\n"
    schema_context += f"Categories (~{len(cats)}): " + ", ".join(cats[:SCHEMA_CAP])
    if len(cats) > SCHEMA_CAP:
        schema_context += "..."
    schema_context += "\n"
    
    schema_context += f"Properties (~{len(props)}): " + ", ".join(props[:SCHEMA_CAP])
    if len(props) > SCHEMA_CAP:
        schema_context += "..."
    schema_context += "\n[END SCHEMA CONTEXT]\n\n"
    
    return schema_context


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

    # -------------------------------------------------------------
    # 0. Check Rate Limit
    # -------------------------------------------------------------
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

    # Track total token usage for this request
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # -------------------------------------------------------------
    # 1. Load or Create Session
    # -------------------------------------------------------------
    db_session: Optional[ChatSession] = None
    history_llm: List[Dict[str, str]] = []
    
    if req.session_id:
        try:
            session_uuid = UUID(req.session_id)
            result = await session.execute(
                select(ChatSession)
                .where(ChatSession.session_id == session_uuid)
                .where(ChatSession.wiki_id == user.wiki_id)
                .where(ChatSession.owner_user_id == user.user_id)
            )
            db_session = result.scalar_one_or_none()
            
            if db_session:
                # Load message history
                history_llm = _convert_db_messages_to_llm_format(db_session.messages)
        except (ValueError, TypeError):
            # Invalid UUID, will create new session
            pass

    # Create new session if needed
    if not db_session:
        db_session = ChatSession(
            wiki_id=user.wiki_id,
            owner_user_id=user.user_id,
        )
        session.add(db_session)
        await session.flush()  # Get the session_id

    current_msgs_llm = _convert_request_messages_to_llm_format(req.messages)
    full_context = history_llm + current_msgs_llm

    # -------------------------------------------------------------
    # 2. Build System Prompt with Schema Context
    # -------------------------------------------------------------
    base_prompt = EDITOR_SYSTEM_PROMPT if req.context == "editor" else CHAT_SYSTEM_PROMPT
    schema_context = await _get_schema_context(vector_store, user.wiki_id)
    system_prompt = base_prompt + schema_context

    # -------------------------------------------------------------
    # 3. LLM + Tool Loop
    # -------------------------------------------------------------
    MAX_LOOPS = 10
    loop_count = 0
    loop_messages = list(full_context)
    used_tools_log: List[Dict[str, str]] = []

    while loop_count < MAX_LOOPS:
        try:
            chat_result = await llm.chat(
                system_prompt,
                loop_messages,
                tools=TOOL_DEFINITIONS,
            )
            response_msg = chat_result.message
            total_prompt_tokens += chat_result.usage.prompt_tokens
            total_completion_tokens += chat_result.usage.completion_tokens
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"LLM call failed (iteration {loop_count}): {str(exc)}",
            ) from exc

        tool_calls = response_msg.get("tool_calls", [])

        if not tool_calls:
            # No tools called -> This is the final answer.
            loop_messages.append(response_msg)
            break
        
        # Tools were called -> Execute them and continue loop
        loop_messages.append(response_msg)
        
        for tc in tool_calls:
            func_name = tc["function"]["name"]
            func_args_str = tc["function"]["arguments"]
            call_id = tc["id"]

            tool_log_entry = {"name": func_name, "args": func_args_str}
            used_tools_log.append(tool_log_entry)

            # Parse arguments
            try:
                parsed_args = json.loads(func_args_str)
            except Exception:
                parsed_args = {}
                tool_result = {"error": "Invalid JSON arguments for tool call."}
                _append_tool_result(loop_messages, call_id, tool_result)
                tool_log_entry["result"] = tool_result
                continue

            # Execute tool
            try:
                tool_output = await dispatch_tool_call(
                    func_name, 
                    parsed_args, 
                    user,
                    vector_store=vector_store,
                    embedder=embedder
                )
            except Exception as exc:
                tool_output = {"error": f"Tool execution failed: {str(exc)}"}

            tool_log_entry["result"] = tool_output
            _append_tool_result(loop_messages, call_id, tool_output)
            
        loop_count += 1

    if loop_count >= MAX_LOOPS:
        # Force one final generation to wrap up
        try:
            chat_result = await llm.chat(
                system_prompt,
                loop_messages,
                tools=TOOL_DEFINITIONS,
            )
            loop_messages.append(chat_result.message)
            total_prompt_tokens += chat_result.usage.prompt_tokens
            total_completion_tokens += chat_result.usage.completion_tokens
        except Exception:
            pass

    # -------------------------------------------------------------
    # 4. Record Token Usage
    # -------------------------------------------------------------
    await rate_limiter.record_usage(
        wiki_id=user.wiki_id,
        user_id=user.user_id,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
    )

    # -------------------------------------------------------------
    # 5. Extract Final Answer
    # -------------------------------------------------------------
    final_answer = loop_messages[-1].get("content", "")

    # -------------------------------------------------------------
    # 6. Persist Messages to Database
    # -------------------------------------------------------------
    # Save user messages
    for msg in req.messages:
        db_msg = ChatMessage(
            session_id=db_session.session_id,
            sender=msg.role,
            content=msg.content,
        )
        session.add(db_msg)

    # Save assistant response
    assistant_msg = ChatMessage(
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
    session.add(assistant_msg)

    # Generate title from first user message if not set
    if not db_session.title and req.messages:
        first_msg = req.messages[0].content[:100]
        db_session.title = first_msg + ("..." if len(req.messages[0].content) > 100 else "")

    # -------------------------------------------------------------
    # 7. Return Response
    # -------------------------------------------------------------
    return ChatResponse(
        messages=req.messages + [ChatMessageModel(role="assistant", content=final_answer)],
        used_tools=used_tools_log,
        session_id=str(db_session.session_id),
    )


# ---------------------------------------------------------------------
# Session Management Routes
# ---------------------------------------------------------------------

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
    """
    List chat sessions for the authenticated user.
    """
    result = await session.execute(
        select(ChatSession)
        .where(ChatSession.wiki_id == user.wiki_id)
        .where(ChatSession.owner_user_id == user.user_id)
        .order_by(ChatSession.updated_at.desc())
        .offset(offset)
        .limit(limit)
    )
    sessions = result.scalars().all()
    
    return [
        {
            "session_id": str(s.session_id),
            "title": s.title,
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "updated_at": s.updated_at.isoformat() if s.updated_at else None,
            "message_count": len(s.messages) if s.messages else 0,
        }
        for s in sessions
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
    """
    Get full conversation history for a session.
    """
    try:
        session_uuid = UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID")

    result = await session.execute(
        select(ChatSession)
        .where(ChatSession.session_id == session_uuid)
        .where(ChatSession.wiki_id == user.wiki_id)
        .where(ChatSession.owner_user_id == user.user_id)
    )
    db_session = result.scalar_one_or_none()
    
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": str(db_session.session_id),
        "title": db_session.title,
        "created_at": db_session.created_at.isoformat() if db_session.created_at else None,
        "updated_at": db_session.updated_at.isoformat() if db_session.updated_at else None,
        "messages": [
            {
                "role": m.sender,
                "content": m.content,
                "created_at": m.created_at.isoformat() if m.created_at else None,
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
    """
    Delete a chat session and all its messages.
    """
    try:
        session_uuid = UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID")

    result = await session.execute(
        select(ChatSession)
        .where(ChatSession.session_id == session_uuid)
        .where(ChatSession.wiki_id == user.wiki_id)
        .where(ChatSession.owner_user_id == user.user_id)
    )
    db_session = result.scalar_one_or_none()
    
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")

    await session.delete(db_session)
    
    return {"status": "deleted", "session_id": session_id}
