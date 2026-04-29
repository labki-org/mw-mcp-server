"""
Async queue for managing background embedding tasks.
"""
import asyncio
import hashlib
import logging
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

from ..config import settings
from ..db import VectorStore, AsyncSessionLocal, Embedding
from .embedder import Embedder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import select

logger = logging.getLogger(__name__)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    length_function=len,
    separators=["\n\n", "\n", ".", " ", ""],
)

@dataclass
class EmbeddingJob:
    """Represents a request to embed a page."""
    wiki_id: str
    title: str
    content: str
    namespace: int
    last_modified: Optional[datetime]
    rev_id: Optional[int] = None

    # Metadata for tracing
    request_id: str = "unknown"

class EmbeddingQueue:
    """Singleton queue for holding embedding jobs."""
    def __init__(self, maxsize: int | None = None):
        effective_size = maxsize if maxsize is not None else settings.embedding_queue_max_size
        self._queue: asyncio.Queue[EmbeddingJob] = asyncio.Queue(maxsize=effective_size)

    async def enqueue(self, job: EmbeddingJob) -> int:
        """Add a job to the queue. Returns current queue size.

        If the queue is full, the oldest job is evicted to make room.
        """
        if self._queue.full():
            try:
                evicted = self._queue.get_nowait()
                logger.warning(f"Queue full ({self._queue.maxsize}), evicted oldest job: {evicted.title}")
                self._queue.task_done()
            except asyncio.QueueEmpty:
                pass  # Shouldn't happen if full() was True, but be safe

        await self._queue.put(job)
        qsize = self._queue.qsize()
        logger.info(f"Job enqueued: {job.title} (Queue size: {qsize})")
        return qsize

    async def get_next_job(self) -> EmbeddingJob:
        return await self._queue.get()

    def task_done(self):
        self._queue.task_done()

# Global singleton
embedding_queue = EmbeddingQueue()


async def process_embeddings_worker_task():
    """
    Background worker that consumes jobs from the queue and manages the embedding process.
    """
    logger.info("Embedding worker started.")
    
    # Instantiate Embedder once (if it holds connections)
    # or create per loop if needed. Here we assume one is fine.
    # Note: VectorStore needs a DB session, so we must create a new session per job.
    embedder = Embedder()

    while True:
        try:
            job = await embedding_queue.get_next_job()
        except asyncio.CancelledError:
            logger.info("Embedding worker cancelled.")
            break

        cancelled = False
        try:
            logger.info(f"Processing embedding job: {job.title} ({job.wiki_id})")
            await _process_single_job(job, embedder)
            logger.info(f"Finished embedding job: {job.title}")
        except asyncio.CancelledError:
            logger.info("Embedding worker cancelled mid-job.")
            cancelled = True
        except Exception:
            logger.exception("Unexpected error in embedding worker")
        finally:
            embedding_queue.task_done()

        if cancelled:
            break

_mismatch_checked: set[str] = set()

async def _check_embedding_model_mismatch(session, wiki_id: str) -> None:
    """Log a warning if any existing embeddings use a different model.

    Caches the result per wiki_id since the configured model is immutable
    at runtime, avoiding a DB query on every job.
    """
    if wiki_id in _mismatch_checked:
        return

    result = await session.execute(
        select(Embedding.embedding_model)
        .where(Embedding.wiki_id == wiki_id)
        .where(Embedding.embedding_model.isnot(None))
        .where(Embedding.embedding_model != settings.embedding_model)
        .limit(1)
    )
    old_model = result.scalar_one_or_none()
    if old_model:
        logger.warning(
            "Embedding model mismatch for wiki %s: existing=%s, current=%s. "
            "Consider re-embedding all pages.",
            wiki_id,
            old_model,
            settings.embedding_model,
        )

    _mismatch_checked.add(wiki_id)


async def _process_single_job(job: EmbeddingJob, embedder: Embedder):
    """
    Execute the embedding logic for a single job inside a dedicated DB session.
    """
    async with AsyncSessionLocal() as session:
        vector_store = VectorStore(session)

        try:
            await _check_embedding_model_mismatch(session, job.wiki_id)

            # Short-circuit if the new content is byte-identical to what we
            # already have indexed under the same model. Saves the OpenAI call
            # and the delete+insert churn for null edits / no-op saves. We
            # still update rev_id / last_modified so the dashboard reads the
            # page as "synced" against the new revision.
            content_sha1 = hashlib.sha1(job.content.encode("utf-8")).hexdigest()
            existing = await vector_store.get_page_sync_state(job.wiki_id, job.title)
            if (
                existing is not None
                and existing.content_sha1 == content_sha1
                and existing.embedding_model == settings.embedding_model
            ):
                await vector_store.touch_page_sync_metadata(
                    wiki_id=job.wiki_id,
                    page_title=job.title,
                    rev_id=job.rev_id,
                    last_modified=job.last_modified,
                )
                await vector_store.commit()
                logger.info(
                    "Embedding unchanged for %s; skipped re-embed (rev_id refreshed).",
                    job.title,
                )
                return

            # 1. Chunk Content
            text_chunks = text_splitter.split_text(job.content)

            # 2. Delete Existing Page Embeddings
            await vector_store.delete_page(job.wiki_id, job.title)

            # Exit if empty content
            if not text_chunks:
                logger.warning(f"No content chunks for {job.title}, skipped.")
                return

            # 3. Embed
            embeddings = await embedder.embed(text_chunks)

            # 4. Add to Index
            section_ids = [f"chunk_{i}" for i in range(len(text_chunks))]
            await vector_store.add_documents(
                wiki_id=job.wiki_id,
                page_titles=[job.title] * len(text_chunks),
                section_ids=section_ids,
                namespaces=[job.namespace] * len(text_chunks),
                embeddings=embeddings,
                last_modified=job.last_modified,
                rev_id=job.rev_id,
                content_sha1=content_sha1,
                embedding_model=settings.embedding_model,
            )

            await vector_store.commit()

        except Exception as e:
            logger.error(f"Failed to process job {job.title}: {e}")
            raise
