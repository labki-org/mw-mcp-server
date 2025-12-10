from fastapi import APIRouter, Depends
from .models import ChatRequest, ChatResponse, ChatMessage
from ..auth.security import verify_mw_to_mcp_jwt, require_scopes
from ..auth.models import UserContext
from ..llm.client import LLMClient
from ..tools import wiki_tools, search_tools
from ..tools.definitions import TOOL_DEFINITIONS
from ..tools.base import dispatch_tool_call
from ..sessions.store import session_store
import json

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/", response_model=ChatResponse)
async def chat(req: ChatRequest, user: UserContext = Depends(require_scopes("chat_completion"))):
    print(f"DEBUG: Incoming Chat Request. Session ID: {req.session_id}, Messages: {len(req.messages)}")
    llm = LLMClient()
    system_prompt = (
        "You are a MediaWiki assistant with access to specific tools to fetch data. "
        "You CANNOT access external websites, but you CAN and should whenever useful "
        "use the provided tools (like mw_get_page) "
        "to retrieve information from the wiki when asked.\n\n"
        "IMPORTANT FORMATTING RULES:\n"
        "1. Always use [[Page Title]] syntax when referring to wiki pages.\n"
        "2. When providing information, try to cite the wiki page source."
    )
    
    # 0. Load History if session_id provided
    history = []
    if req.session_id:
        history = session_store.get_history(req.session_id)
        # Convert Pydantic models to dicts for LLM
        history_dicts = [{"role": m.role, "content": m.content} for m in history]
    else:
        history_dicts = []
    
    # 1. Initialize messages (History + Current Request)
    current_request_msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    full_context = history_dicts + current_request_msgs
    
    # 2. First call to LLM
    response_msg = await llm.chat(
        system_prompt, 
        full_context, 
        tools=TOOL_DEFINITIONS
    )
    
    # Track conversation for THIS request only (for tool loop)
    # We copy full_context to track the tool loop, but we won't save intermediate tool calls to long-term history yet
    # (simplification: only save User query and Final Answer)
    loop_messages = list(full_context) 
    
    used_tools_log = []
    
    # 3. Check for tool calls
    if response_msg.get("tool_calls"):
        # Append assistant's "thought" (function call request) to history
        loop_messages.append(response_msg)
        
        tool_calls = response_msg["tool_calls"]
        for tc in tool_calls:
            func_name = tc["function"]["name"]
            func_args_str = tc["function"]["arguments"]
            call_id = tc["id"]
            
            used_tools_log.append({"name": func_name, "args": func_args_str})
            
            try:
                args = json.loads(func_args_str)
                result = await dispatch_tool_call(func_name, args, user)
                result_str = json.dumps(result, default=str)
            except Exception as e:
                result_str = f"Error executing tool: {str(e)}"

            # Append tool result to history
            loop_messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": result_str
            })
            
        # 4. Follow-up call to LLM with tool outputs
        final_msg = await llm.chat(
            system_prompt,
            loop_messages,
            tools=TOOL_DEFINITIONS
        )
        loop_messages.append(final_msg)
    else:
        loop_messages.append(response_msg)

    final_content = loop_messages[-1].get("content") or ""
    
    # 5. Save to Session Store (User's new messages + Assistant's final answer)
    if req.session_id:
        new_history_items = req.messages + [ChatMessage(role="assistant", content=final_content)]
        session_store.add_messages(req.session_id, new_history_items)
    
    return ChatResponse(
        messages=req.messages + [ChatMessage(role="assistant", content=final_content)],
        used_tools=used_tools_log,
    )
