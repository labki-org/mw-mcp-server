"""Add content_sha1 column to embedding table

Revision ID: 0004
Revises: 0003
Create Date: 2026-04-29

When the worker is asked to re-embed a page, it can short-circuit the OpenAI
call entirely if the new content's SHA1 matches what we already have stored.
This catches the common case of null edits / no-op saves that bump rev_id
without changing content. Nullable so existing rows keep working under the
"no hash known, must re-embed" fallback path.
"""
from typing import Sequence, Union

from alembic import op


revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE embedding ADD COLUMN IF NOT EXISTS content_sha1 VARCHAR(40)"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE embedding DROP COLUMN IF EXISTS content_sha1")
