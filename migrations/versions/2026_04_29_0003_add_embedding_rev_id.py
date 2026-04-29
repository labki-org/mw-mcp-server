"""Add rev_id column to embedding table

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-29

The MediaWiki extension previously sent only `last_modified` (a timestamp), which
the dashboard compared against `page_touched`. `page_touched` is bumped by
parser-cache invalidation events (template edits, link recounts, SMW reparses)
that don't change page content, so embeddings appeared "outdated" after normal
background work. Tracking `rev_id` lets us compare against `page_latest`
directly — equality is then a true identity check.

Nullable + IF NOT EXISTS so this is safe to re-run and so existing rows keep
working under fallback timestamp comparison until they're naturally re-embedded.
ADD COLUMN with no default is metadata-only in PostgreSQL >= 11, so no rewrite.
"""
from typing import Sequence, Union

from alembic import op


revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE embedding ADD COLUMN IF NOT EXISTS rev_id BIGINT"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE embedding DROP COLUMN IF EXISTS rev_id")
