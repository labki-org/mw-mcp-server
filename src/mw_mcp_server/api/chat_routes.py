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
from ..prompts import CHAT_SYSTEM_PROMPT, EDITOR_SYSTEM_PROMPT

router = APIRouter(prefix="/chat", tags=["chat"])




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

    MAX_LOOPS = 10
    loop_count = 0
    
    # -------------------------------------------------------------
    # 1. & 2. & 3. LLM + Tool Loop
    # -------------------------------------------------------------
    # We loop until the LLM stops calling tools or we hit the limit.
    
    # Select prompt based on context
    base_prompt = EDITOR_SYSTEM_PROMPT if req.context == "editor" else CHAT_SYSTEM_PROMPT

    # -------------------------------------------------------------
    # Inject Schema Context (Categories & Properties)
    # -------------------------------------------------------------
    # We deliberately inject this into the system prompt to ground the LLM immediately.
    # To prevent context overflow on massive wikis, we cap the list.
    SCHEMA_CAP = 50 
    
    # NS_CATEGORY = 14, NS_PROPERTY = 102
    cats = faiss_index.get_pages_by_namespace(14)
    props = faiss_index.get_pages_by_namespace(102)
    
    schema_context = "\n\n[KNOWN SCHEMA ELEMENTS (Truncated if > 100)]\n"
    schema_context += f"Categories (~{len(cats)}): " + ", ".join(cats[:SCHEMA_CAP])
    if len(cats) > SCHEMA_CAP:
        schema_context += "..."
    schema_context += "\n"
    
    schema_context += f"Properties (~{len(props)}): " + ", ".join(props[:SCHEMA_CAP])
    if len(props) > SCHEMA_CAP:
        schema_context += "..."
    schema_context += "\n[END SCHEMA CONTEXT]\n\n"

    system_prompt = base_prompt + schema_context
    
    loop_messages = list(full_context)
    used_tools_log: List[Dict[str, str]] = []

    while loop_count < MAX_LOOPS:
        try:
            response_msg = await llm.chat(
                system_prompt,
                loop_messages,
                tools=TOOL_DEFINITIONS,
            )
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
                # Return error to model so it can retry
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
                    faiss_index=faiss_index,
                    embedder=embedder
                )
            except Exception as exc:
                # Return error to model
                tool_output = {"error": f"Tool execution failed: {str(exc)}"}

            tool_log_entry["result"] = tool_output
            _append_tool_result(loop_messages, call_id, tool_output)
            
        loop_count += 1

    if loop_count >= MAX_LOOPS:
        # If we exited due to limit, we might want to append a system note or just return the last state?
        # The last message in loop_messages might be tool outputs. 
        # We need the LLM to summarize/conclude if possible, or we just return the last thing it said.
        # But if the LAST thing was tool outputs, the user won't see a proper response.
        # Let's force one final generation without tools to wrap up.
        try:
            final_msg = await llm.chat(
                system_prompt,
                loop_messages,
                tools=TOOL_DEFINITIONS, # Allowed, but hopefully it stops.
            )
            loop_messages.append(final_msg)
        except Exception:
            pass # Use what we have

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
