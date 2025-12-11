"""
Embedding Data Models

This module defines the canonical data model used to represent a single
indexed document chunk stored in the FAISS vector index.

Each instance corresponds to ONE embedding vector and ONE chunk of text.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class IndexedDocument(BaseModel):
    """
    A single indexed document chunk.

    This model is the authoritative schema for:
    - FAISS index storage
    - Metadata persistence to JSON
    - Vector search result mapping
    """

    page_title: str = Field(
        ...,
        min_length=1,
        description="Canonical MediaWiki page title this chunk belongs to.",
    )

    section_id: Optional[str] = Field(
        default=None,
        description="Optional section identifier within the page.",
    )

    text: str = Field(
        ...,
        min_length=1,
        description="Raw text content for this embedded chunk.",
    )

    namespace: int = Field(
        ...,
        ge=0,
        description="MediaWiki namespace ID for permission filtering.",
    )

    last_modified: Optional[str] = Field(
        default=None,
        description="ISO 8601 timestamp for the page revision this chunk came from.",
    )

    model_config = ConfigDict(
        extra="forbid",          # Prevent schema injection
        frozen=True,            # Make instances immutable once created
        arbitrary_types_allowed=False,
    )
