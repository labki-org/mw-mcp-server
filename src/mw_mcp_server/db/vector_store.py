"""
Vector Store

PostgreSQL + pgvector based vector storage and similarity search.
Replaces the previous FAISS-based implementation.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
from datetime import datetime

from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Embedding


class VectorStore:
    """
    PostgreSQL-backed vector store using pgvector for similarity search.
    
    This class provides the same interface as the old FaissIndex but
    uses the database for persistence.
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
                embedding=emb,
            )
            self._session.add(embedding_record)

        await self._session.flush()
        return len(embeddings)

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
        # Build the cosine similarity query using pgvector's <=> operator
        cosine_distance = Embedding.embedding.cosine_distance(query_embedding)
        
        stmt = (
            select(
                Embedding.page_title,
                Embedding.section_id,
                Embedding.namespace,
                (1 - cosine_distance).label("score"),
            )
            .where(Embedding.wiki_id == wiki_id)
            .order_by(cosine_distance)
            .limit(k)
        )

        if namespace_filter:
            stmt = stmt.where(Embedding.namespace.in_(namespace_filter))

        result = await self._session.execute(stmt)
        rows = result.all()

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

        # Unique pages
        pages_stmt = (
            select(Embedding.page_title)
            .where(Embedding.wiki_id == wiki_id)
            .distinct()
        )
        pages_result = await self._session.execute(pages_stmt)
        unique_pages = sorted([row[0] for row in pages_result.all()])

        return {
            "total_vectors": total_vectors,
            "total_pages": len(unique_pages),
            "embedded_pages": unique_pages,
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
