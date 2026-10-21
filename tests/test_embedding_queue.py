"""
Embedding Queue Tests

The queue is backed by the ``embedding_job`` table, so most of these run
against a real PostgreSQL (skipped when unavailable). Covers per-page dedupe,
back-pressure, retries, the blank-page path and the delete/update race.
"""

import asyncio
import hashlib

import httpx
import pytest
from sqlalchemy import func, select

from mw_mcp_server.config import settings
from mw_mcp_server.db import Embedding, EmbeddingJobRecord, VectorStore
from mw_mcp_server.embeddings import queue as queue_module
from mw_mcp_server.embeddings.embedder import EmbeddingError, _is_transient_http_error
from mw_mcp_server.embeddings.queue import (
    JOB_OP_DELETE,
    EmbeddingJob,
    EmbeddingQueue,
    QueueFullError,
    _process_single_job,
    backoff_seconds,
    describe_error,
    process_embeddings_worker_task,
)

from pg_utils import pg_session_factory_fixture  # noqa: F401  (registers fixture)

WIKI = "test-wiki"


def _make_job(title: str, content: str = "test content", **kw) -> EmbeddingJob:
    return EmbeddingJob(wiki_id=WIKI, title=title, content=content, namespace=0, **kw)


class FakeEmbedder:
    """Returns constant vectors; optionally runs a hook mid-embed."""

    def __init__(self, during_embed=None, exc=None):
        self.calls = 0
        self.during_embed = during_embed
        self.exc = exc

    async def embed(self, texts):
        self.calls += 1
        if self.during_embed is not None:
            await self.during_embed()
        if self.exc is not None:
            raise self.exc
        return [[0.01] * settings.embedding_dimensions for _ in texts]


async def _embedding_rows(sf, title):
    async with sf() as s:
        return (
            await s.execute(
                select(Embedding.content_sha1).where(
                    Embedding.wiki_id == WIKI, Embedding.page_title == title
                )
            )
        ).scalars().all()


async def _job_rows(sf):
    async with sf() as s:
        return (await s.execute(select(EmbeddingJobRecord))).scalars().all()


async def _run_one(queue, embedder):
    claimed = await queue.claim_next()
    assert claimed is not None
    await _process_single_job(claimed, embedder, queue)
    return claimed


# ---------------------------------------------------------------------
# Pure unit tests
# ---------------------------------------------------------------------

def _override_settings(monkeypatch, **overrides):
    """Override optional queue knobs read through ``queue._setting``."""
    original = queue_module._setting
    monkeypatch.setattr(
        queue_module,
        "_setting",
        lambda name, default: overrides.get(name, original(name, default)),
    )


def test_backoff_is_exponential_and_capped(monkeypatch):
    _override_settings(
        monkeypatch,
        embedding_job_backoff_base_seconds=2.0,
        embedding_job_backoff_max_seconds=60.0,
    )
    assert 1.6 <= backoff_seconds(1) <= 2.4
    assert 6.4 <= backoff_seconds(3) <= 9.6
    assert backoff_seconds(30) <= 72.0  # cap * max jitter


def test_transient_http_errors_are_retryable():
    req = httpx.Request("POST", "https://api.openai.com/v1/embeddings")

    def status_error(code):
        return httpx.HTTPStatusError("x", request=req, response=httpx.Response(code, request=req))

    assert _is_transient_http_error(status_error(429))
    assert _is_transient_http_error(status_error(503))
    assert _is_transient_http_error(httpx.ReadTimeout("t", request=req))
    assert not _is_transient_http_error(status_error(400))
    assert not _is_transient_http_error(status_error(401))


def test_describe_error_never_includes_content():
    from sqlalchemy.exc import IntegrityError

    secret = "SECRET PAGE CONTENT"
    db_exc = IntegrityError("INSERT ...", {"content": secret}, ValueError(secret))
    assert secret not in describe_error(db_exc)
    assert describe_error(db_exc) == "IntegrityError(ValueError)"

    req = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    resp = httpx.Response(429, request=req, text=secret)
    http_exc = httpx.HTTPStatusError(secret, request=req, response=resp)
    wrapped = EmbeddingError("Embedding generation failed", retryable=True)
    wrapped.__cause__ = http_exc
    assert describe_error(wrapped) == "EmbeddingError(HTTP 429)"
    assert describe_error(RuntimeError(secret)) == "RuntimeError"


# ---------------------------------------------------------------------
# DB-backed tests
# ---------------------------------------------------------------------

async def test_enqueue_dedupes_by_page_keeping_latest(pg_session_factory):
    q = EmbeddingQueue(maxsize=10, session_factory=pg_session_factory)
    assert await q.enqueue(_make_job("A", "v1")) == 1
    assert await q.enqueue(_make_job("A", "v2")) == 1
    assert await q.enqueue(_make_job("B")) == 2

    rows = {r.page_title: r for r in await _job_rows(pg_session_factory)}
    assert set(rows) == {"A", "B"}
    assert rows["A"].content == "v2"
    assert rows["A"].generation == 2


async def test_full_queue_rejects_new_pages_but_accepts_updates(pg_session_factory):
    q = EmbeddingQueue(maxsize=2, session_factory=pg_session_factory)
    await q.enqueue(_make_job("A"))
    await q.enqueue(_make_job("B"))

    with pytest.raises(QueueFullError):
        await q.enqueue(_make_job("C"))

    # Replacing a pending page doesn't grow the queue, so it's accepted.
    assert await q.enqueue(_make_job("A", "newer")) == 2
    assert {r.page_title for r in await _job_rows(pg_session_factory)} == {"A", "B"}


async def test_queue_cap_is_per_wiki(pg_session_factory):
    """One wiki filling its share must not block another wiki."""
    q = EmbeddingQueue(maxsize=2, global_maxsize=100, session_factory=pg_session_factory)
    await q.enqueue(_make_job("A"))
    await q.enqueue(_make_job("B"))
    with pytest.raises(QueueFullError):
        await q.enqueue(_make_job("C"))

    other = EmbeddingJob(wiki_id="other-wiki", title="X", content="x")
    assert await q.enqueue(other) == 1


async def test_global_safety_cap(pg_session_factory):
    q = EmbeddingQueue(maxsize=10, global_maxsize=2, session_factory=pg_session_factory)
    await q.enqueue(_make_job("A"))
    await q.enqueue(EmbeddingJob(wiki_id="other-wiki", title="X", content="x"))
    with pytest.raises(QueueFullError):
        await q.enqueue(EmbeddingJob(wiki_id="third-wiki", title="Y", content="y"))


async def test_claims_round_robin_across_wikis(pg_session_factory):
    q = EmbeddingQueue(maxsize=100, session_factory=pg_session_factory)
    for i in range(5):
        await q.enqueue(_make_job(f"Big{i}"))
    await q.enqueue(EmbeddingJob(wiki_id="small-wiki", title="S", content="s"))

    first = await q.claim_next()
    second = await q.claim_next()
    # The small wiki is served right after the first big-wiki job, not after
    # the whole backlog.
    assert {first.job.wiki_id, second.job.wiki_id} == {WIKI, "small-wiki"}


async def test_worker_embeds_page_and_removes_job(pg_session_factory):
    q = EmbeddingQueue(maxsize=10, session_factory=pg_session_factory)
    await q.enqueue(_make_job("A", "hello world"))

    await _run_one(q, FakeEmbedder())

    sha = hashlib.sha1(b"hello world").hexdigest()
    assert await _embedding_rows(pg_session_factory, "A") == [sha]
    assert await _job_rows(pg_session_factory) == []


async def test_unchanged_content_skips_embedding_call(pg_session_factory):
    q = EmbeddingQueue(maxsize=10, session_factory=pg_session_factory)
    await q.enqueue(_make_job("A", "same"))
    await _run_one(q, FakeEmbedder())

    await q.enqueue(_make_job("A", "same", rev_id=7))
    embedder = FakeEmbedder()
    await _run_one(q, embedder)

    assert embedder.calls == 0
    assert await _job_rows(pg_session_factory) == []


async def test_blanking_page_removes_existing_embeddings(pg_session_factory):
    """Empty content must commit the delete (it used to be rolled back)."""
    q = EmbeddingQueue(maxsize=10, session_factory=pg_session_factory)
    await q.enqueue(_make_job("A", "some text"))
    await _run_one(q, FakeEmbedder())
    assert len(await _embedding_rows(pg_session_factory, "A")) == 1

    await q.enqueue(_make_job("A", ""))
    await _run_one(q, FakeEmbedder())

    assert await _embedding_rows(pg_session_factory, "A") == []
    assert await _job_rows(pg_session_factory) == []


async def test_delete_during_inflight_update_is_not_resurrected(pg_session_factory):
    """DELETE arriving while an older update is being embedded wins."""
    q = EmbeddingQueue(maxsize=10, session_factory=pg_session_factory)
    await q.enqueue(_make_job("A", "old content"))

    async def delete_now():
        async with pg_session_factory() as s:
            await VectorStore(s).delete_page_and_pending_job(WIKI, "A")
            await s.commit()

    await _run_one(q, FakeEmbedder(during_embed=delete_now))

    assert await _embedding_rows(pg_session_factory, "A") == []
    assert await _job_rows(pg_session_factory) == []


async def test_delete_cancels_queued_update(pg_session_factory):
    q = EmbeddingQueue(maxsize=10, session_factory=pg_session_factory)
    await q.enqueue(_make_job("A", "queued but not started"))

    async with pg_session_factory() as s:
        await VectorStore(s).delete_page_and_pending_job(WIKI, "A")
        await s.commit()

    assert await q.claim_next() is None
    assert await _embedding_rows(pg_session_factory, "A") == []


async def test_queued_delete_op_removes_embeddings(pg_session_factory):
    q = EmbeddingQueue(maxsize=10, session_factory=pg_session_factory)
    await q.enqueue(_make_job("A", "text"))
    await _run_one(q, FakeEmbedder())

    await q.enqueue(EmbeddingJob(wiki_id=WIKI, title="A", op=JOB_OP_DELETE))
    await _run_one(q, FakeEmbedder())

    assert await _embedding_rows(pg_session_factory, "A") == []


async def test_update_during_inflight_update_processes_latest(pg_session_factory):
    q = EmbeddingQueue(maxsize=10, session_factory=pg_session_factory)
    await q.enqueue(_make_job("A", "v1"))

    async def requeue():
        await q.enqueue(_make_job("A", "v2"))

    await _run_one(q, FakeEmbedder(during_embed=requeue))
    # v1 result discarded; v2 still pending and immediately claimable.
    assert await _embedding_rows(pg_session_factory, "A") == []
    claimed = await _run_one(q, FakeEmbedder())
    assert claimed.job.content == "v2"
    assert await _embedding_rows(pg_session_factory, "A") == [
        hashlib.sha1(b"v2").hexdigest()
    ]


async def test_retryable_failure_is_rescheduled_with_backoff(pg_session_factory):
    q = EmbeddingQueue(maxsize=10, session_factory=pg_session_factory)
    await q.enqueue(_make_job("A"))
    claimed = await q.claim_next()

    exc = EmbeddingError("Embedding generation failed", retryable=True)
    with pytest.raises(EmbeddingError):
        await _process_single_job(claimed, FakeEmbedder(exc=exc), q)
    await q.record_failure(claimed, exc)

    [row] = await _job_rows(pg_session_factory)
    assert row.attempts == 1
    assert row.leased_until is None
    assert row.last_error == "EmbeddingError"
    async with pg_session_factory() as s:
        assert row.next_attempt_at > (await s.execute(select(func.now()))).scalar()
    # Not due yet.
    assert await q.claim_next() is None


async def test_permanent_failure_drops_job(pg_session_factory):
    q = EmbeddingQueue(maxsize=10, session_factory=pg_session_factory)
    await q.enqueue(_make_job("A"))
    claimed = await q.claim_next()
    await q.record_failure(claimed, EmbeddingError("400", retryable=False))
    assert await _job_rows(pg_session_factory) == []


async def test_retries_are_bounded(pg_session_factory, monkeypatch):
    _override_settings(monkeypatch, embedding_job_max_attempts=2)
    q = EmbeddingQueue(maxsize=10, session_factory=pg_session_factory)
    await q.enqueue(_make_job("A"))

    claimed = await q.claim_next()
    await q.record_failure(claimed, EmbeddingError("503", retryable=True))
    assert len(await _job_rows(pg_session_factory)) == 1

    claimed.attempts = 1  # as if re-claimed after the backoff elapsed
    await q.record_failure(claimed, EmbeddingError("503", retryable=True))
    assert await _job_rows(pg_session_factory) == []


async def test_lease_prevents_double_claim(pg_session_factory):
    q = EmbeddingQueue(maxsize=10, session_factory=pg_session_factory)
    await q.enqueue(_make_job("A"))
    assert await q.claim_next() is not None
    assert await q.claim_next() is None


async def test_worker_task_drains_queue(pg_session_factory, monkeypatch):
    _override_settings(monkeypatch, embedding_worker_poll_seconds=0.05)
    q = EmbeddingQueue(maxsize=10, session_factory=pg_session_factory)
    task = asyncio.create_task(process_embeddings_worker_task(q, FakeEmbedder()))
    try:
        await q.enqueue(_make_job("A", "alpha"))
        await q.enqueue(_make_job("B", "beta"))
        for _ in range(100):
            if not await _job_rows(pg_session_factory):
                break
            await asyncio.sleep(0.05)
        assert await _job_rows(pg_session_factory) == []
        assert len(await _embedding_rows(pg_session_factory, "B")) == 1
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
