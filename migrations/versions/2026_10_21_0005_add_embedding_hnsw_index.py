"""Add HNSW index for cosine similarity search on embeddings

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-21

Vector search previously did an exact sequential scan over every row. pgvector's
HNSW index on the ``vector`` type is capped at 2000 dimensions, and the default
model (text-embedding-3-large) produces 3072, so the index is built on an
expression that casts to ``halfvec`` (capped at 4000 dims, needs pgvector
>= 0.7.0). ``VectorStore.search`` orders by exactly the same expression with
the cosine operator (``<=>``), which is what lets the planner use it.

The dimension baked into the expression comes from EMBEDDING_DIMENSIONS at
migration time. If that setting is ever changed, the column type changes too
and this index has to be rebuilt along with a full re-embed.

Runs ``ALTER EXTENSION vector UPDATE`` first: volumes created by an older image
keep the old extension version after the binary is upgraded. Fails with a clear
message if the library is still older than 0.7.0.

Built CONCURRENTLY so the embedding worker can keep writing during the build.
Note: on a large existing table the build runs during startup (the app applies
migrations before serving); raise maintenance_work_mem for big corpora.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from mw_mcp_server.config import settings


revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

INDEX_NAME = "idx_embedding_hnsw_cosine"
HALFVEC_MAX_DIMS = 4000
MIN_PGVECTOR = (0, 7, 0)


def _version_tuple(version) -> tuple:
    try:
        return tuple(int(p) for p in str(version).split(".")[:3])
    except ValueError:
        return (0,)


def upgrade() -> None:
    dims = settings.embedding_dimensions
    if dims > HALFVEC_MAX_DIMS:
        # halfvec HNSW can't index this; search falls back to an exact scan.
        return

    with op.get_context().autocommit_block():
        # Volumes created with an older image keep the old extension version
        # even after the server binary is upgraded; bring it up to the
        # installed library version, then require halfvec support.
        op.execute("ALTER EXTENSION vector UPDATE")
        version = op.get_bind().execute(
            sa.text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar()
        if _version_tuple(version) < MIN_PGVECTOR:
            raise RuntimeError(
                f"pgvector {version} is installed but halfvec HNSW indexes need "
                ">= 0.7.0. Upgrade the postgres image (pgvector/pgvector:0.8.x-pg16) "
                "and re-run migrations."
            )

        # A failed CONCURRENTLY build leaves an INVALID index behind that
        # IF NOT EXISTS would silently keep; drop it so we rebuild cleanly.
        op.execute(
            f"""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM pg_index i
                    JOIN pg_class c ON c.oid = i.indexrelid
                    WHERE c.relname = '{INDEX_NAME}' AND NOT i.indisvalid
                ) THEN
                    EXECUTE 'DROP INDEX {INDEX_NAME}';
                END IF;
            END $$;
            """
        )
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {INDEX_NAME} "
            f"ON embedding USING hnsw ((embedding::halfvec({dims})) halfvec_cosine_ops)"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
