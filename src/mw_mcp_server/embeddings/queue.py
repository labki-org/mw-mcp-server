"""
Async queue for managing background embedding tasks.
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

from ..db import VectorStore, AsyncSessionLocal
from .embedder import Embedder
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Re-use the same splitter configuration
# (Ideally this would be injected or shared config, but duplicating for simplicity here)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=12000,
    chunk_overlap=1200,
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
    
    # Metadata for tracing
    request_id: str = "unknown"

class EmbeddingQueue:
    """Singleton queue for holding embedding jobs."""
    def __init__(self):
        self._queue: asyncio.Queue[EmbeddingJob] = asyncio.Queue()

    async def enqueue(self, job: EmbeddingJob) -> int:
        """Add a job to the queue. Returns current queue size."""
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
            logger.info(f"Processing embedding job: {job.title} ({job.wiki_id})")

            await _process_single_job(job, embedder)
            
            embedding_queue.task_done()
            logger.info(f"Finished embedding job: {job.title}")

        except asyncio.CancelledError:
            logger.info("Embedding worker cancelled.")
            break
        except Exception:
            logger.exception("Unexpected error in embedding worker")
            # Don't break the loop on random errors
            continue

async def _process_single_job(job: EmbeddingJob, embedder: Embedder):
    """
    Execute the embedding logic for a single job inside a dedicated DB session.
    """
    async with AsyncSessionLocal() as session:
        vector_store = VectorStore(session)
        
        try:
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
            )
            
            await vector_store.commit()
            
        except Exception as e:
            logger.error(f"Failed to process job {job.title}: {e}")
            raise
