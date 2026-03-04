"""Add embedding_model column to embedding table

Revision ID: 0001
Revises:
Create Date: 2026-03-04

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '0001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add embedding_model column to track which model generated each embedding."""
    # IF NOT EXISTS makes this idempotent — safe if create_all already added it
    op.execute(
        "ALTER TABLE embedding ADD COLUMN IF NOT EXISTS "
        "embedding_model VARCHAR(128)"
    )


def downgrade() -> None:
    """Remove embedding_model column."""
    op.drop_column('embedding', 'embedding_model')
