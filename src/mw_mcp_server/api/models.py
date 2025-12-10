from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    session_id: Optional[str] = None
    max_tokens: Optional[int] = 512
    tools_mode: str = "auto"

class ChatResponse(BaseModel):
    messages: List[ChatMessage]
    used_tools: List[Dict[str, Any]] = []

class SearchRequest(BaseModel):
    query: str
    k: int = 5

class SearchResult(BaseModel):
    title: str
    section_id: str | None
    score: float
    text: str

class SMWQueryRequest(BaseModel):
    ask: str

class SMWQueryResponse(BaseModel):
    raw: dict

class EditRequest(BaseModel):
    title: str
    new_text: str
    summary: str = "AI-assisted edit"

class EditResponse(BaseModel):
    success: bool
    new_rev_id: int | None

class EmbeddingUpdatePageRequest(BaseModel):
    title: str
    content: str
    last_modified: str | None = None

class EmbeddingDeletePageRequest(BaseModel):
    title: str

class EmbeddingStatsResponse(BaseModel):
    total_vectors: int
    total_pages: int
    embedded_pages: List[str]
    page_timestamps: Dict[str, str] = {}
