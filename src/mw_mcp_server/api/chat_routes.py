from fastapi import APIRouter, Depends
from .models import ChatRequest, ChatResponse, ChatMessage
from ..auth.security import get_current_user
from ..llm.client import LLMClient
from ..tools import wiki_tools, search_tools

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/", response_model=ChatResponse)
async def chat(req: ChatRequest, user = Depends(get_current_user)):
    # 1. Decide if tools needed (simple heuristic or LLM).
    # 2. Call tools if necessary.
    # 3. Call LLM with context and tool outputs.
    llm = LLMClient()
    system_prompt = "You are an assistant answering using the wiki's content."
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    answer = await llm.chat(system_prompt, messages)
    return ChatResponse(
        messages=req.messages + [ChatMessage(role="assistant", content=answer)],
        used_tools=[],
    )
