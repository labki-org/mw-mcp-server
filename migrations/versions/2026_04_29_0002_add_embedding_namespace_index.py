"""Add (wiki_id, namespace) index to embedding table

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-29

The schema-context lookup runs `SELECT DISTINCT page_title WHERE wiki_id=? AND namespace=?`
on every chat request. The existing (wiki_id, page_title) index doesn't help that filter,
so this migration adds (wiki_id, namespace, page_title) — covering for the DISTINCT scan.
"""
from typing import Sequence, Union

from alembic import op


revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_embedding_wiki_ns "
        "ON embedding (wiki_id, namespace, page_title)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_embedding_wiki_ns")
