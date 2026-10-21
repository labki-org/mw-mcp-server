"""Base schema (embedding, chat_session, chat_message, token_usage)

Revision ID: 0000
Revises:
Create Date: 2026-03-01

Historically the base tables were only ever created by ``Base.metadata.create_all``
at application startup, and revision 0001 assumed they already existed. That made
a bare ``alembic upgrade head`` against an empty database fail at 0001
("relation embedding does not exist").

This revision creates the schema as it stood *before* 0001 so the migration chain
is self-contained. Every table is guarded with an existence check, so databases
that were bootstrapped by ``create_all`` (or that are already stamped at a later
revision) are unaffected: the app still runs ``create_all`` before ``upgrade``,
and alembic never re-runs a revision that is already below the stamped head.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID
from pgvector.sqlalchemy import Vector

from mw_mcp_server.config import settings


revision: str = "0000"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def upgrade() -> None:
    """Create the pre-0001 base tables if they don't already exist."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    if not _has_table("embedding"):
        op.create_table(
            "embedding",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("wiki_id", sa.String(64), nullable=False),
            sa.Column("page_title", sa.Text(), nullable=False),
            sa.Column("section_id", sa.Text(), nullable=True),
            sa.Column("namespace", sa.Integer(), nullable=False),
            sa.Column("last_modified", sa.DateTime(), nullable=True),
            sa.Column("embedding", Vector(settings.embedding_dimensions), nullable=False),
        )
        op.create_index("ix_embedding_wiki_id", "embedding", ["wiki_id"])
        op.create_index("idx_embedding_wiki_page", "embedding", ["wiki_id", "page_title"])

    if not _has_table("chat_session"):
        op.create_table(
            "chat_session",
            sa.Column("session_id", UUID(as_uuid=True), primary_key=True),
            sa.Column("wiki_id", sa.String(64), nullable=False),
            sa.Column("owner_user_id", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("title", sa.Text(), nullable=True),
            sa.Column("summary", sa.Text(), nullable=True),
        )
        op.create_index("idx_session_owner", "chat_session", ["wiki_id", "owner_user_id"])

    if not _has_table("chat_message"):
        op.create_table(
            "chat_message",
            sa.Column("message_id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column(
                "session_id",
                UUID(as_uuid=True),
                sa.ForeignKey("chat_session.session_id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("sender", sa.String(16), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("metadata", JSONB(), nullable=True),
        )
        op.create_index("idx_message_session", "chat_message", ["session_id", "created_at"])

    if not _has_table("token_usage"):
        op.create_table(
            "token_usage",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("wiki_id", sa.String(64), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("usage_date", sa.Date(), nullable=False),
            sa.Column("prompt_tokens", sa.Integer(), nullable=False),
            sa.Column("completion_tokens", sa.Integer(), nullable=False),
            sa.Column("total_tokens", sa.Integer(), nullable=False),
            sa.Column("request_count", sa.Integer(), nullable=False),
            sa.UniqueConstraint("wiki_id", "user_id", "usage_date", name="uq_usage_user_date"),
        )
        op.create_index(
            "idx_usage_lookup", "token_usage", ["wiki_id", "user_id", "usage_date"]
        )


def downgrade() -> None:
    """Intentionally a no-op: dropping every table is never what you want here."""
    pass
