"""Add embedding_job table (durable, per-page-deduplicated embedding queue)

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-21

Embedding updates used to live in an in-process asyncio.Queue: lost on restart,
oldest silently evicted when full, no retry, and deletes bypassed it entirely
(so an older queued update could re-insert a page right after it was deleted).
This table holds at most one pending operation per (wiki_id, page_title).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # create_all may already have created it at startup.
    if sa.inspect(op.get_bind()).has_table("embedding_job"):
        return

    op.create_table(
        "embedding_job",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("wiki_id", sa.String(64), nullable=False),
        sa.Column("page_title", sa.Text(), nullable=False),
        sa.Column("op", sa.String(16), nullable=False),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("namespace", sa.Integer(), nullable=True),
        sa.Column("last_modified", sa.DateTime(), nullable=True),
        sa.Column("rev_id", sa.BigInteger(), nullable=True),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "next_attempt_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("leased_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("wiki_id", "page_title", name="uq_embedding_job_page"),
    )
    op.create_index("idx_embedding_job_due", "embedding_job", ["next_attempt_at"])


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS embedding_job")
