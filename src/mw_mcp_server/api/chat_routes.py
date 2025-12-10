"""
Chat Routes: LLM-Driven Conversational Interface

This module implements the core conversational endpoint used by the MediaWiki
extension. It provides:
- Multi-turn chat session support backed by a session store
- Tool invocation loops based on LLM function calls
- Secure, scope-controlled access with full user context
- Structured output suitable for the MediaWiki client

Major Responsibilities
----------------------
1. Accept ChatRequest containing user messages and optional session_id.
2. Retrieve prior session history (if applicable).
3. Call the LLM with:
    - System prompt
    - Combined history + new user messages
    - Tool definitions
4. Detect and execute LLM tool/function calls.
5. Make a second LLM call with the tool results.
6. Save final messages back to the session store.
7. Return ChatResponse with final assistant message + tool usage log.

Security Model
--------------
- The route requires the `chat_completion` scope.
- Authentication via JWT is handled upstream by the `require_scopes` dependency.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated, List, Dict, Any
import json

from .models import ChatRequest, ChatResponse, ChatMessage
from ..auth.models import UserContext
from ..auth.security import require_scopes
from ..llm.client import LLMClient
from ..embeddings.index import FaissIndex
from ..embeddings.embedder import Embedder
from ..tools.definitions import TOOL_DEFINITIONS
from ..tools.base import dispatch_tool_call
from ..sessions.store import session_store
from .dependencies import get_llm_client, get_faiss_index, get_embedder

router = APIRouter(prefix="/chat", tags=["chat"])

# ---------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a MediaWiki assistant with access to specific tools to fetch data. "
    "You cannot access external websites, but you SHOULD use the available tools "
    "whenever they help answer a user's question.\n\n"
    "RULES:\n"
    "1. Always use [[Page Title]] format when referring to wiki pages.\n"
    "2. Cite sources by naming the wiki page your info comes from.\n"
)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def _convert_history_to_llm_format(history: List[ChatMessage]) -> List[Dict[str, str]]:
    """Convert ChatMessage objects into plain dicts for LLM input."""
    return [{"role": m.role, "content": m.content} for m in history]


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
    faiss_index: Annotated[FaissIndex, Depends(get_faiss_index)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
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
    # 0. Load Session History
    # -------------------------------------------------------------
    if req.session_id:
        history = session_store.get_history(req.session_id)
        history_llm = _convert_history_to_llm_format(history)
    else:
        history = []
        history_llm = []

    current_msgs_llm = _convert_history_to_llm_format(req.messages)
    full_context = history_llm + current_msgs_llm

    # -------------------------------------------------------------
    # 1. Initial LLM Call
    # -------------------------------------------------------------
    # llm injected via dependency

    try:
        response_msg = await llm.chat(
            SYSTEM_PROMPT,
            full_context,
            tools=TOOL_DEFINITIONS,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM call failed: {str(exc)}",
        ) from exc

    loop_messages = list(full_context)
    used_tools_log: List[Dict[str, str]] = []

    # -------------------------------------------------------------
    # 2. Tool Loop
    # -------------------------------------------------------------
    tool_calls = response_msg.get("tool_calls", [])

    if tool_calls:
        # Add the assistant tool-call message
        loop_messages.append(response_msg)

        for tc in tool_calls:
            func_name = tc["function"]["name"]
            func_args_str = tc["function"]["arguments"]
            call_id = tc["id"]

            used_tools_log.append({"name": func_name, "args": func_args_str})

            # Parse arguments
            try:
                parsed_args = json.loads(func_args_str)
            except Exception:
                parsed_args = {}
                tool_result = {"error": "Invalid JSON arguments for tool call."}
                _append_tool_result(loop_messages, call_id, tool_result)
                continue

            # Execute tool
            try:
                tool_output = await dispatch_tool_call(
                    func_name, 
                    parsed_args, 
                    user,
                    faiss_index=faiss_index,
                    embedder=embedder
                )
            except Exception as exc:
                tool_output = {"error": f"Tool execution failed: {str(exc)}"}

            _append_tool_result(loop_messages, call_id, tool_output)

        # ---------------------------------------------------------
        # 3. Follow-up LLM Call
        # ---------------------------------------------------------
        try:
            final_msg = await llm.chat(
                SYSTEM_PROMPT,
                loop_messages,
                tools=TOOL_DEFINITIONS,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"LLM follow-up call failed: {str(exc)}",
            ) from exc

        loop_messages.append(final_msg)
    else:
        # No tool calls — direct assistant response
        loop_messages.append(response_msg)

    # -------------------------------------------------------------
    # 4. Extract Final Answer
    # -------------------------------------------------------------
    final_answer = loop_messages[-1].get("content", "")

    # -------------------------------------------------------------
    # 5. Save New Conversation State
    # -------------------------------------------------------------
    if req.session_id:
        session_store.add_messages(
            req.session_id,
            req.messages + [ChatMessage(role="assistant", content=final_answer)],
        )

    # -------------------------------------------------------------
    # 6. Return Response
    # -------------------------------------------------------------
    return ChatResponse(
        messages=req.messages + [ChatMessage(role="assistant", content=final_answer)],
        used_tools=used_tools_log,
    )
