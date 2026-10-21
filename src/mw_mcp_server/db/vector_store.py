"""
Vector Store

PostgreSQL + pgvector based vector storage and similarity search.
Replaces the previous FAISS-based implementation.
"""

from __future__ import annotations

import logging
from typing import List, NamedTuple, Tuple, Optional
from datetime import datetime

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import cast, delete, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Embedding
from ..config import settings

logger = logging.getLogger(__name__)

# halfvec HNSW indexes support up to 4000 dimensions (pgvector >= 0.7.0).
HALFVEC_MAX_DIMS = 4000

# HNSW applies the WHERE clause (wiki_id / namespace) *after* the index scan,
# so with the default ef_search=40 a small wiki in a shared table can get
# fewer than k rows back. pgvector >= 0.8 can keep scanning until enough rows
# pass the filter (hnsw.iterative_scan); older versions only get a larger
# ef_search. Detected once per process.
_HNSW_EF_SEARCH_MIN = 100
_iterative_scan_supported: Optional[bool] = None


class PageSyncState(NamedTuple):
    """What we know about a page's currently-stored embedding."""
    content_sha1: Optional[str]
    rev_id: Optional[int]
    embedding_model: Optional[str]


def _version_at_least(version: Optional[str], minimum: Tuple[int, ...]) -> bool:
    if not version:
        return False
    try:
        parts = tuple(int(p) for p in version.split(".")[: len(minimum)])
    except ValueError:
        return False
    return parts >= minimum


class VectorStore:
    """
    PostgreSQL-backed vector store using pgvector for similarity search.
    
    Provides vector storage and similarity search backed by PostgreSQL + pgvector.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize with an async database session.
        
        Parameters
        ----------
        session : AsyncSession
            SQLAlchemy async session for database operations.
        """
        self._session = session

    async def commit(self) -> None:
        """
        Commit the current transaction.
        """
        await self._session.commit()

    async def add_documents(
        self,
        wiki_id: str,
        page_titles: List[str],
        section_ids: List[Optional[str]],
        namespaces: List[int],
        embeddings: List[List[float]],
        last_modified: Optional[datetime] = None,
        rev_id: Optional[int] = None,
        content_sha1: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> int:
        """
        Add document embeddings to the store.

        Parameters
        ----------
        wiki_id : str
            Tenant wiki identifier.
        page_titles : List[str]
            Page titles for each embedding.
        section_ids : List[Optional[str]]
            Section identifiers (can be None).
        namespaces : List[int]
            MediaWiki namespace IDs.
        embeddings : List[List[float]]
            Vector embeddings matching the page_titles.
        last_modified : Optional[datetime]
            Timestamp for the embeddings.
        rev_id : Optional[int]
            MediaWiki revision ID this content was taken from.
        embedding_model : Optional[str]
            Name of the model used to generate these embeddings.

        Returns
        -------
        int
            Number of embeddings added.
        """
        if not embeddings:
            return 0

        for i, (title, section, ns, emb) in enumerate(
            zip(page_titles, section_ids, namespaces, embeddings)
        ):
            embedding_record = Embedding(
                wiki_id=wiki_id,
                page_title=title,
                section_id=section,
                namespace=ns,
                last_modified=last_modified,
                rev_id=rev_id,
                content_sha1=content_sha1,
                embedding=emb,
                embedding_model=embedding_model,
            )
            self._session.add(embedding_record)

        await self._session.flush()
        return len(embeddings)

    async def get_page_sync_state(
        self,
        wiki_id: str,
        page_title: str,
    ) -> Optional[PageSyncState]:
        """
        Return the sync metadata for the most recent embedding row of
        *page_title*, or None if the page has never been embedded. Reads a
        single row (any chunk) since these fields are identical across chunks
        of the same indexed revision.
        """
        stmt = (
            select(
                Embedding.content_sha1,
                Embedding.rev_id,
                Embedding.embedding_model,
            )
            .where(Embedding.wiki_id == wiki_id)
            .where(Embedding.page_title == page_title)
            .limit(1)
        )
        result = await self._session.execute(stmt)
        row = result.first()
        if row is None:
            return None
        return PageSyncState(
            content_sha1=row.content_sha1,
            rev_id=row.rev_id,
            embedding_model=row.embedding_model,
        )

    async def touch_page_sync_metadata(
        self,
        wiki_id: str,
        page_title: str,
        rev_id: Optional[int],
        last_modified: Optional[datetime],
    ) -> int:
        """
        Update sync metadata (rev_id, last_modified) on every chunk of a page
        without touching the vectors themselves. Used to record that a null edit
        bumped rev_id even though content is unchanged, so the dashboard reads
        the page as "synced" without paying for re-embedding.
        """
        values: dict = {}
        if rev_id is not None:
            values["rev_id"] = rev_id
        if last_modified is not None:
            values["last_modified"] = last_modified
        if not values:
            return 0

        stmt = (
            update(Embedding)
            .where(Embedding.wiki_id == wiki_id)
            .where(Embedding.page_title == page_title)
            .values(**values)
        )
        result = await self._session.execute(stmt)
        return result.rowcount or 0

    async def delete_page(self, wiki_id: str, page_title: str) -> int:
        """
        Remove all embeddings for a given page.
        
        Returns the number of deleted rows.
        """
        stmt = delete(Embedding).where(
            Embedding.wiki_id == wiki_id,
            Embedding.page_title == page_title,
        )
        result = await self._session.execute(stmt)
        return result.rowcount

    async def _configure_hnsw_scan(self, k: int) -> None:
        """Tune HNSW for filtered queries for the rest of this transaction."""
        global _iterative_scan_supported
        if _iterative_scan_supported is None:
            version = (
                await self._session.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                )
            ).scalar()
            _iterative_scan_supported = _version_at_least(version, (0, 8))
            if not _iterative_scan_supported:
                logger.warning(
                    "pgvector %s lacks hnsw.iterative_scan (needs >= 0.8.0); filtered "
                    "vector searches may return fewer than k results.",
                    version,
                )

        ef_search = max(_HNSW_EF_SEARCH_MIN, k * 4)
        # SET LOCAL can't take bind parameters; ef_search is an int we computed.
        await self._session.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
        if _iterative_scan_supported:
            await self._session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))

    async def search(
        self,
        wiki_id: str,
        query_embedding: List[float],
        k: int = 5,
        namespace_filter: Optional[List[int]] = None,
    ) -> List[Tuple[str, Optional[str], int, float]]:
        """
        Search for similar documents using cosine similarity.
        
        Parameters
        ----------
        wiki_id : str
            Tenant wiki identifier.
        query_embedding : List[float]
            Query vector.
        k : int
            Number of results to return.
        namespace_filter : Optional[List[int]]
            If provided, only return results from these namespaces.
            
        Returns
        -------
        List[Tuple[str, Optional[str], int, float]]
            List of (page_title, section_id, namespace, score) tuples.
        """
        dims = settings.embedding_dimensions
        exact_distance = Embedding.embedding.cosine_distance(query_embedding)

        if dims > HALFVEC_MAX_DIMS:
            # No HNSW index is possible; exact scan on the raw vector column.
            order_distance = exact_distance
        else:
            # Must match the indexed expression in migration 0005 exactly
            # (``embedding::halfvec(N)`` with ``<=>``) or the planner won't
            # use the HNSW index.
            order_distance = cast(Embedding.embedding, HALFVEC(dims)).cosine_distance(
                cast(query_embedding, HALFVEC(dims))
            )
            await self._configure_hnsw_scan(k)

        stmt = (
            select(
                Embedding.page_title,
                Embedding.section_id,
                Embedding.namespace,
                # Report the full-precision score; only the ordering uses halfvec.
                (1 - exact_distance).label("score"),
            )
            .where(Embedding.wiki_id == wiki_id)
            .order_by(order_distance)
            .limit(k)
        )

        if namespace_filter:
            stmt = stmt.where(Embedding.namespace.in_(namespace_filter))

        result = await self._session.execute(stmt)
        # relaxed_order iterative scans may return slightly out-of-order rows.
        rows = sorted(result.all(), key=lambda r: r.score, reverse=True)

        return [(row.page_title, row.section_id, row.namespace, row.score) for row in rows]

    async def get_pages_by_namespace(
        self,
        wiki_id: str,
        namespace: Optional[int] = None,
        pattern: Optional[str] = None,
    ) -> List[str]:
        """
        Return distinct page titles, optionally filtered by namespace and pattern.
        """
        stmt = select(Embedding.page_title).where(Embedding.wiki_id == wiki_id).distinct()

        if namespace is not None:
            stmt = stmt.where(Embedding.namespace == namespace)

        if pattern:
            stmt = stmt.where(Embedding.page_title.ilike(f"%{pattern}%"))

        result = await self._session.execute(stmt)
        return sorted([row[0] for row in result.all()])

    async def get_embedding_last_modified(
        self,
        wiki_id: str,
        page_title: Optional[str] = None,
    ) -> Optional[datetime]:
        """
        Return the most recent last_modified timestamp for embeddings.

        If page_title is given, scopes to that page. Otherwise returns the
        latest timestamp across all embeddings for the wiki.
        """
        stmt = (
            select(func.max(Embedding.last_modified))
            .where(Embedding.wiki_id == wiki_id)
        )
        if page_title is not None:
            stmt = stmt.where(Embedding.page_title == page_title)
        result = await self._session.execute(stmt)
        return result.scalar()

    async def get_stats(self, wiki_id: str) -> dict:
        """
        Return statistics about the vector store for a wiki.
        """
        # Total vectors
        total_stmt = select(func.count()).select_from(Embedding).where(
            Embedding.wiki_id == wiki_id
        )
        total_result = await self._session.execute(total_stmt)
        total_vectors = total_result.scalar() or 0

        # Unique pages with their latest timestamp and rev_id. Group by title
        # so multi-chunk pages collapse into one row.
        pages_stmt = (
            select(
                Embedding.page_title,
                func.max(Embedding.last_modified),
                func.max(Embedding.rev_id),
            )
            .where(Embedding.wiki_id == wiki_id)
            .group_by(Embedding.page_title)
        )
        pages_result = await self._session.execute(pages_stmt)

        embedded_pages = []
        page_timestamps = {}
        page_revisions = {}

        for row in pages_result.all():
            title = row[0]
            last_mod = row[1]
            rev_id = row[2]
            embedded_pages.append(title)
            if last_mod:
                # Return MediaWiki-compatible format: YYYYMMDDHHMMSS
                page_timestamps[title] = last_mod.strftime("%Y%m%d%H%M%S")
            if rev_id is not None:
                page_revisions[title] = int(rev_id)

        return {
            "total_vectors": total_vectors,
            "total_pages": len(embedded_pages),
            "embedded_pages": sorted(embedded_pages),
            "page_timestamps": page_timestamps,
            "page_revisions": page_revisions,
        }

    async def rebuild(
        self,
        wiki_id: str,
        page_titles: List[str],
        section_ids: List[Optional[str]],
        namespaces: List[int],
        embeddings: List[List[float]],
        last_modified: Optional[datetime] = None,
    ) -> int:
        """
        Delete all embeddings for a wiki and replace with new ones.
        """
        # Delete all existing
        delete_stmt = delete(Embedding).where(Embedding.wiki_id == wiki_id)
        await self._session.execute(delete_stmt)

        # Add new
        return await self.add_documents(
            wiki_id=wiki_id,
            page_titles=page_titles,
            section_ids=section_ids,
            namespaces=namespaces,
            embeddings=embeddings,
            last_modified=last_modified,
        )
