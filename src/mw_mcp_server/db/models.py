"""
SQLAlchemy Models

Defines the database schema for:
- Embeddings (vector storage with pgvector)
- Chat sessions and messages
- Token usage tracking for rate limiting
"""

from __future__ import annotations

import uuid
from datetime import datetime, date
from typing import Optional, List

from sqlalchemy import (
    Column,
    String,
    Integer,
    Text,
    DateTime,
    Date,
    ForeignKey,
    Index,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


# ---------------------------------------------------------------------
# Embedding Model (replaces FAISS)
# ---------------------------------------------------------------------

class Embedding(Base):
    """
    Vector embedding for wiki page chunks.
    
    Uses pgvector for similarity search.
    """
    __tablename__ = "embedding"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    wiki_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    page_title: Mapped[str] = mapped_column(Text, nullable=False)
    section_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    namespace: Mapped[int] = mapped_column(Integer, nullable=False)
    last_modified: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # pgvector column - 3072 dimensions for text-embedding-3-large
    embedding = Column(Vector(3072), nullable=False)

    __table_args__ = (
        Index("idx_embedding_wiki_page", "wiki_id", "page_title"),
    )


# ---------------------------------------------------------------------
# Chat Session Model
# ---------------------------------------------------------------------

class ChatSession(Base):
    """
    A chat conversation session owned by a user.
    """
    __tablename__ = "chat_session"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    wiki_id: Mapped[str] = mapped_column(String(64), nullable=False)
    owner_user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationship to messages
    messages: Mapped[List["ChatMessage"]] = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )

    __table_args__ = (
        Index("idx_session_owner", "wiki_id", "owner_user_id"),
    )


# ---------------------------------------------------------------------
# Chat Message Model
# ---------------------------------------------------------------------

class ChatMessage(Base):
    """
    A single message within a chat session.
    """
    __tablename__ = "chat_message"

    message_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_session.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    sender: Mapped[str] = mapped_column(String(16), nullable=False)  # user | assistant | tool
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[Optional[dict]] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
    )

    # Relationship back to session
    session: Mapped["ChatSession"] = relationship("ChatSession", back_populates="messages")

    __table_args__ = (
        Index("idx_message_session", "session_id", "created_at"),
    )


# ---------------------------------------------------------------------
# Token Usage Model (Rate Limiting)
# ---------------------------------------------------------------------

class TokenUsage(Base):
    """
    Daily token usage record per user per wiki.
    
    Tracks prompt and completion tokens consumed by LLM calls
    for rate limiting purposes.
    """
    __tablename__ = "token_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    wiki_id: Mapped[str] = mapped_column(String(64), nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    usage_date: Mapped[date] = mapped_column(Date, nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    request_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    __table_args__ = (
        UniqueConstraint("wiki_id", "user_id", "usage_date", name="uq_usage_user_date"),
        Index("idx_usage_lookup", "wiki_id", "user_id", "usage_date"),
    )

