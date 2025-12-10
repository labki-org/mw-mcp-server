"""
API Models for MCP Server

This module defines all Pydantic models used for request/response validation
across chat, search, editing, and embedding-related endpoints.

Design Goals
------------
- Strong typing
- Safe defaults (no shared mutable state)
- Clear schema documentation
- Forward compatibility with testing and OpenAPI generation
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


# --- Tool Output Contracts ---

class ToolSearchResult(BaseModel):
    title: str
    section_id: str | None
    score: float
    text: str

class OperationResult(BaseModel):
    status: Literal["updated", "deleted", "created", "ok"]
    count: int | None = None
    details: Dict[str, Any] | None = None


# ---------------------------------------------------------------------
# Chat Models
# ---------------------------------------------------------------------

class ChatMessage(BaseModel):
    """
    Single message in a chat conversation.
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    """
    Chat completion request payload.
    """
    messages: List[ChatMessage] = Field(..., min_length=1)
    session_id: Optional[str] = None
    max_tokens: Optional[int] = Field(default=512, ge=1, le=32768)
    tools_mode: Literal["auto", "none", "forced"] = "auto"


class ChatResponse(BaseModel):
    """
    Chat completion response payload.
    """
    messages: List[ChatMessage]
    used_tools: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------
# Search Models
# ---------------------------------------------------------------------

class SearchRequest(BaseModel):
    """
    Vector or text-based search request.
    """
    query: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=100)


class SearchResult(BaseModel):
    """
    Individual search match.
    """
    title: str
    section_id: Optional[str] = None
    score: float
    text: str


# ---------------------------------------------------------------------
# Semantic MediaWiki (SMW) Query Models
# ---------------------------------------------------------------------

class SMWQueryRequest(BaseModel):
    """
    Raw SMW ask query wrapper.
    """
    ask: str = Field(..., min_length=1)


class SMWQueryResponse(BaseModel):
    """
    Raw SMW ask response wrapper.
    """
    raw: Dict[str, Any]


# ---------------------------------------------------------------------
# Edit Models
# ---------------------------------------------------------------------

class EditRequest(BaseModel):
    """
    Page edit request issued by the LLM or MediaWiki client.
    """
    title: str = Field(..., min_length=1)
    new_text: str = Field(..., min_length=1)
    summary: str = Field(default="AI-assisted edit", min_length=1)


class EditResponse(BaseModel):
    """
    Result of a page edit operation.
    """
    success: bool
    new_rev_id: Optional[int] = None


# ---------------------------------------------------------------------
# Embedding Models
# ---------------------------------------------------------------------

class EmbeddingUpdatePageRequest(BaseModel):
    """
    Request to (re)index a wiki page for embedding search.
    """
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    last_modified: Optional[str] = None


class EmbeddingDeletePageRequest(BaseModel):
    """
    Request to delete all embeddings for a given page.
    """
    title: str = Field(..., min_length=1)


class EmbeddingStatsResponse(BaseModel):
    """
    Statistics for the embedding index.
    """
    total_vectors: int = Field(..., ge=0)
    total_pages: int = Field(..., ge=0)
    embedded_pages: List[str] = Field(default_factory=list)
    page_timestamps: Dict[str, str] = Field(default_factory=dict)
