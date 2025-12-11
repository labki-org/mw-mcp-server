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
- Explicit tool output contracts
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------
# Tool Output Contracts (Authoritative)
# ---------------------------------------------------------------------

class ToolSearchResult(BaseModel):
    """
    Canonical tool-layer result for vector search.
    This model defines the exact output contract for:
        - tools/search_tools.py
        - LLM tool responses
        - API search routes
    """
    title: str = Field(..., min_length=1)
    section_id: Optional[str] = None
    score: float = Field(..., ge=0.0)
    text: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class OperationResult(BaseModel):
    """
    Standardized mutation operation result.
    Used for create/update/delete-style endpoints.
    """
    status: Literal["updated", "deleted", "created", "ok"]
    count: Optional[int] = Field(default=None, ge=0)
    details: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------
# Chat Models
# ---------------------------------------------------------------------

class ChatMessage(BaseModel):
    """
    Single message in a chat conversation.
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(..., min_length=1)

    model_config = ConfigDict(extra="forbid")


class ChatRequest(BaseModel):
    """
    Chat completion request payload.
    """
    messages: List[ChatMessage] = Field(..., min_length=1)
    session_id: Optional[str] = None
    max_tokens: Optional[int] = Field(default=512, ge=1, le=32768)
    tools_mode: Literal["auto", "none", "forced"] = "auto"
    context: Literal["chat", "editor"] = "chat"

    model_config = ConfigDict(extra="forbid")


class ChatResponse(BaseModel):
    """
    Chat completion response payload.
    """
    messages: List[ChatMessage]
    used_tools: List[Dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------
# Search Models
# ---------------------------------------------------------------------

class SearchRequest(BaseModel):
    """
    Vector or text-based search request.
    """
    query: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=100)

    model_config = ConfigDict(extra="forbid")


class SearchResult(BaseModel):
    """
    Individual search match.
    """
    title: str = Field(..., min_length=1)
    section_id: Optional[str] = None
    score: float = Field(..., ge=0.0)
    text: str = Field(..., min_length=1)

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------
# Semantic MediaWiki (SMW) Query Models
# ---------------------------------------------------------------------

class SMWQueryRequest(BaseModel):
    """
    Raw SMW ask query wrapper.
    """
    ask: str = Field(..., min_length=1)

    model_config = ConfigDict(extra="forbid")


class SMWQueryResponse(BaseModel):
    """
    Raw SMW ask response wrapper.
    """
    raw: Dict[str, Any]

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------
# Embedding Models
# ---------------------------------------------------------------------

class EmbeddingUpdatePageRequest(BaseModel):
    """
    Request to (re)index a wiki page for embedding search.
    """
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    namespace: int = Field(default=0, ge=0)
    last_modified: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class EmbeddingDeletePageRequest(BaseModel):
    """
    Request to delete all embeddings for a given page.
    """
    title: str = Field(..., min_length=1)

    model_config = ConfigDict(extra="forbid")


class EmbeddingStatsResponse(BaseModel):
    """
    Statistics for the embedding index.
    """
    total_vectors: int = Field(..., ge=0)
    total_pages: int = Field(..., ge=0)
    embedded_pages: List[str] = Field(default_factory=list)
    page_timestamps: Dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")
