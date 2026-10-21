"""
Durable queue for background embedding jobs.

Jobs live in the ``embedding_job`` table rather than in process memory, so
pending work survives restarts and can be shared by several server processes.

Guarantees
----------
- **Per-page dedupe, latest wins.** There is at most one pending row per
  (wiki_id, title). Enqueuing a page that is already pending overwrites the
  payload and bumps ``generation``; only the newest content is ever embedded.
- **No resurrection after delete.** ``VectorStore.delete_page_and_pending_job``
  removes the pending row and the vectors in one transaction. Before writing,
  the worker re-locks its row and checks the generation; if the page was
  deleted (row gone) or re-queued (generation changed) it discards its result.
- **Bounded retries.** Transient failures (OpenAI 429/5xx/timeouts, DB
  hiccups) are retried with capped exponential backoff, then dropped with an
  error log. The wiki can re-embed via Special:MWAssistantEmbeddings.
- **Back-pressure instead of silent loss.** When the queue is at capacity a
  *new* page is rejected with ``QueueFullError`` (HTTP 503 at the route) so
  the wiki job retries later; updates to an already-pending page are always
  accepted since they don't grow the queue.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..db import AsyncSessionLocal, Embedding, EmbeddingJobRecord, VectorStore
from .embedder import Embedder, EmbeddingError

logger = logging.getLogger(__name__)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    length_function=len,
    separators=["\n\n", "\n", ".", " ", ""],
)

JOB_OP_UPSERT = "upsert"
JOB_OP_DELETE = "delete"


def _setting(name: str, default):
    """Read an optional tuning knob that may not be declared in Settings yet."""
    return getattr(settings, name, default)


class QueueFullError(RuntimeError):
    """Raised when a new page can't be queued because the queue is at capacity."""


@dataclass
class EmbeddingJob:
    """Represents a request to embed (or delete) a page."""
    wiki_id: str
    title: str
    content: str = ""
    namespace: int = 0
    last_modified: Optional[datetime] = None
    rev_id: Optional[int] = None
    op: str = JOB_OP_UPSERT

    # Metadata for tracing
    request_id: str = "unknown"


@dataclass
class ClaimedJob:
    """A job leased by a worker, with the generation it was leased at."""
    id: int
    generation: int
    attempts: int
    job: EmbeddingJob


SessionFactory = Callable[[], AsyncSession]


class EmbeddingQueue:
    """PostgreSQL-backed embedding work queue (see module docstring)."""

    def __init__(
        self,
        maxsize: int | None = None,
        session_factory: Optional[SessionFactory] = None,
    ):
        self.maxsize = maxsize if maxsize is not None else settings.embedding_queue_max_size
        self._session_factory = session_factory or AsyncSessionLocal
        self._wakeup: Optional[asyncio.Event] = None

    # -----------------------------------------------------------------
    # Producer side
    # -----------------------------------------------------------------

    async def enqueue(self, job: EmbeddingJob) -> int:
        """Queue (or replace) the pending operation for a page.

        Returns the number of pending jobs after the call.

        Raises
        ------
        QueueFullError
            If the page isn't already pending and the queue is at capacity.
        """
        async with self._session_factory() as session:
            already_pending = (
                await session.execute(
                    select(EmbeddingJobRecord.id)
                    .where(EmbeddingJobRecord.wiki_id == job.wiki_id)
                    .where(EmbeddingJobRecord.page_title == job.title)
                )
            ).scalar_one_or_none() is not None
            pending = (
                await session.execute(select(func.count()).select_from(EmbeddingJobRecord))
            ).scalar_one()

            # The capacity check is advisory (concurrent enqueues can overshoot
            # by a few rows); its purpose is back-pressure, not a hard bound.
            if not already_pending and pending >= self.maxsize:
                logger.warning(
                    "Embedding queue full (%d/%d); rejecting %s:%s",
                    pending, self.maxsize, job.wiki_id, job.title,
                )
                raise QueueFullError(f"Embedding queue is full ({pending} pending)")

            values = dict(
                op=job.op,
                content=job.content if job.op == JOB_OP_UPSERT else None,
                namespace=job.namespace,
                last_modified=job.last_modified,
                rev_id=job.rev_id,
            )
            stmt = pg_insert(EmbeddingJobRecord).values(
                wiki_id=job.wiki_id,
                page_title=job.title,
                generation=1,
                attempts=0,
                **values,
            )
            stmt = stmt.on_conflict_do_update(
                constraint="uq_embedding_job_page",
                set_={
                    **values,
                    "generation": EmbeddingJobRecord.generation + 1,
                    "attempts": 0,
                    "last_error": None,
                    "next_attempt_at": func.now(),
                    "updated_at": func.now(),
                    # leased_until is deliberately left alone: a worker that is
                    # mid-job keeps its lease and will notice the new generation.
                },
            )
            await session.execute(stmt)
            await session.commit()

        self._notify()
        size = pending if already_pending else pending + 1
        logger.info(f"Job enqueued: {job.title} (Queue size: {size})")
        return size

    async def pending_count(self) -> int:
        async with self._session_factory() as session:
            return (
                await session.execute(select(func.count()).select_from(EmbeddingJobRecord))
            ).scalar_one()

    # -----------------------------------------------------------------
    # Consumer side
    # -----------------------------------------------------------------

    async def claim_next(self) -> Optional[ClaimedJob]:
        """Lease the next due job, or return None if nothing is due."""
        lease = timedelta(seconds=_setting("embedding_job_lease_seconds", 600))
        async with self._session_factory() as session:
            due = (
                select(EmbeddingJobRecord.id)
                .where(EmbeddingJobRecord.next_attempt_at <= func.now())
                .where(
                    (EmbeddingJobRecord.leased_until.is_(None))
                    | (EmbeddingJobRecord.leased_until < func.now())
                )
                .order_by(EmbeddingJobRecord.next_attempt_at, EmbeddingJobRecord.id)
                .limit(1)
                .with_for_update(skip_locked=True)
                .scalar_subquery()
            )
            row = (
                await session.execute(
                    update(EmbeddingJobRecord)
                    .where(EmbeddingJobRecord.id == due)
                    .values(leased_until=func.now() + lease)
                    .returning(EmbeddingJobRecord)
                )
            ).scalar_one_or_none()
            await session.commit()

        if row is None:
            return None
        return ClaimedJob(
            id=row.id,
            generation=row.generation,
            attempts=row.attempts,
            job=EmbeddingJob(
                wiki_id=row.wiki_id,
                title=row.page_title,
                content=row.content or "",
                namespace=row.namespace or 0,
                last_modified=row.last_modified,
                rev_id=row.rev_id,
                op=row.op,
                request_id=f"job-{row.id}-g{row.generation}",
            ),
        )

    async def lock_if_current(self, session: AsyncSession, claimed: ClaimedJob) -> bool:
        """Row-lock the job in *session*'s transaction if it is still current.

        Returns False if the page was deleted (row gone) or re-queued
        (generation changed) since the job was claimed; the caller must then
        discard its work. Holding the lock until commit serialises the worker's
        write against concurrent enqueue/delete for the same page.
        """
        generation = (
            await session.execute(
                select(EmbeddingJobRecord.generation)
                .where(EmbeddingJobRecord.id == claimed.id)
                .with_for_update()
            )
        ).scalar_one_or_none()
        return generation == claimed.generation

    async def finish_locked(self, session: AsyncSession, claimed: ClaimedJob) -> None:
        """Remove a job row previously locked via ``lock_if_current``."""
        await session.execute(
            delete(EmbeddingJobRecord).where(EmbeddingJobRecord.id == claimed.id)
        )

    async def release(self, claimed: ClaimedJob) -> None:
        """Drop the lease so the (possibly newer) row is picked up again."""
        async with self._session_factory() as session:
            await session.execute(
                update(EmbeddingJobRecord)
                .where(EmbeddingJobRecord.id == claimed.id)
                .values(leased_until=None)
            )
            await session.commit()
        self._notify()

    async def record_failure(self, claimed: ClaimedJob, exc: BaseException) -> None:
        """Schedule a retry with exponential backoff, or drop the job."""
        max_attempts = _setting("embedding_job_max_attempts", 6)
        attempts = claimed.attempts + 1
        retryable = _is_retryable(exc)
        error = f"{type(exc).__name__}: {exc}"[:2000]

        async with self._session_factory() as session:
            superseded = not await self.lock_if_current(session, claimed)
            if superseded:
                # A newer payload (with a fresh attempt budget) replaced this
                # one; just let it be picked up.
                await session.execute(
                    update(EmbeddingJobRecord)
                    .where(EmbeddingJobRecord.id == claimed.id)
                    .values(leased_until=None)
                )
            elif not retryable or attempts >= max_attempts:
                logger.error(
                    "Dropping embedding job %s:%s after %d attempt(s) (%s): %s",
                    claimed.job.wiki_id, claimed.job.title, attempts,
                    "retryable" if retryable else "permanent", error,
                )
                await self.finish_locked(session, claimed)
            else:
                delay = backoff_seconds(attempts)
                logger.warning(
                    "Embedding job %s:%s failed (attempt %d/%d); retrying in %.0fs: %s",
                    claimed.job.wiki_id, claimed.job.title, attempts, max_attempts,
                    delay, error,
                )
                await session.execute(
                    update(EmbeddingJobRecord)
                    .where(EmbeddingJobRecord.id == claimed.id)
                    .values(
                        attempts=attempts,
                        last_error=error,
                        leased_until=None,
                        next_attempt_at=func.now() + timedelta(seconds=delay),
                    )
                )
            await session.commit()
        if superseded:
            self._notify()

    async def wait_for_work(self, timeout: float) -> None:
        """Sleep until something is enqueued in this process or *timeout* passes."""
        event = self._event()
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        event.clear()

    def _event(self) -> asyncio.Event:
        if self._wakeup is None:
            self._wakeup = asyncio.Event()
        return self._wakeup

    def _notify(self) -> None:
        if self._wakeup is not None:
            self._wakeup.set()


def backoff_seconds(attempt: int) -> float:
    """Capped exponential backoff with +/-20% jitter for retry *attempt* (1-based)."""
    base = float(_setting("embedding_job_backoff_base_seconds", 5.0))
    cap = float(_setting("embedding_job_backoff_max_seconds", 900.0))
    delay = min(cap, base * (2 ** max(0, attempt - 1)))
    return delay * random.uniform(0.8, 1.2)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, EmbeddingError):
        return exc.retryable
    # DB connectivity errors etc. are worth retrying; attempts are bounded.
    return True


# Global singleton
embedding_queue = EmbeddingQueue()


async def process_embeddings_worker_task(
    queue: Optional[EmbeddingQueue] = None,
    embedder: Optional[Embedder] = None,
):
    """
    Background worker that leases jobs from the queue and processes them.
    """
    logger.info("Embedding worker started.")
    queue = queue or embedding_queue
    embedder = embedder or Embedder()
    poll_seconds = float(_setting("embedding_worker_poll_seconds", 5.0))

    while True:
        try:
            claimed = await queue.claim_next()
        except asyncio.CancelledError:
            logger.info("Embedding worker cancelled.")
            break
        except Exception:
            logger.exception("Failed to claim embedding job; backing off")
            try:
                await asyncio.sleep(poll_seconds)
            except asyncio.CancelledError:
                break
            continue

        if claimed is None:
            try:
                await queue.wait_for_work(poll_seconds)
            except asyncio.CancelledError:
                logger.info("Embedding worker cancelled.")
                break
            continue

        job = claimed.job
        try:
            logger.info(f"Processing embedding job: {job.title} ({job.wiki_id}, {job.op})")
            await _process_single_job(claimed, embedder, queue)
            logger.info(f"Finished embedding job: {job.title}")
        except asyncio.CancelledError:
            logger.info("Embedding worker cancelled mid-job; releasing lease.")
            try:
                await asyncio.shield(queue.release(claimed))
            except BaseException:
                pass  # Lease expiry will make it claimable again.
            break
        except Exception as exc:
            try:
                await queue.record_failure(claimed, exc)
            except Exception:
                logger.exception("Failed to record embedding job failure for %s", job.title)


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


async def _process_single_job(
    claimed: ClaimedJob,
    embedder: Embedder,
    queue: Optional[EmbeddingQueue] = None,
) -> None:
    """
    Execute one leased job. The vector writes and the removal of the job row
    commit atomically, and only if the job is still the current one for its
    page; otherwise the work is discarded and the newer job runs instead.
    """
    queue = queue or embedding_queue
    job = claimed.job

    async with queue._session_factory() as session:
        vector_store = VectorStore(session)

        if job.op == JOB_OP_DELETE:
            if not await queue.lock_if_current(session, claimed):
                await session.rollback()
                return
            await vector_store.delete_page(job.wiki_id, job.title)
            await queue.finish_locked(session, claimed)
            await session.commit()
            return

        await _check_embedding_model_mismatch(session, job.wiki_id)

        # Short-circuit if the new content is byte-identical to what we
        # already have indexed under the same model. Saves the OpenAI call
        # and the delete+insert churn for null edits / no-op saves. We
        # still update rev_id / last_modified so the dashboard reads the
        # page as "synced" against the new revision.
        content_sha1 = hashlib.sha1(job.content.encode("utf-8")).hexdigest()
        existing = await vector_store.get_page_sync_state(job.wiki_id, job.title)
        # End the read transaction before the (slow) embedding call so we
        # don't sit idle-in-transaction holding a pooled connection's snapshot.
        await session.commit()

        unchanged = (
            existing is not None
            and existing.content_sha1 == content_sha1
            and existing.embedding_model == settings.embedding_model
        )

        text_chunks: list[str] = []
        embeddings: list[list[float]] = []
        if not unchanged:
            text_chunks = text_splitter.split_text(job.content)
            if text_chunks:
                embeddings = await embedder.embed(text_chunks)

        if not await queue.lock_if_current(session, claimed):
            logger.info("Embedding job for %s superseded; discarding result.", job.title)
            await session.rollback()
            await queue.release(claimed)
            return

        if unchanged:
            await vector_store.touch_page_sync_metadata(
                wiki_id=job.wiki_id,
                page_title=job.title,
                rev_id=job.rev_id,
                last_modified=job.last_modified,
            )
            logger.info(
                "Embedding unchanged for %s; skipped re-embed (rev_id refreshed).",
                job.title,
            )
        else:
            await vector_store.delete_page(job.wiki_id, job.title)
            if text_chunks:
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
            else:
                # Blanked page: the delete above must still be committed.
                logger.warning(f"No content chunks for {job.title}; removed its embeddings.")

        await queue.finish_locked(session, claimed)
        await session.commit()
