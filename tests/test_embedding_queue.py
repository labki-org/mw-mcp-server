"""
Embedding Queue Tests

Tests for the bounded embedding queue with eviction logic.
"""

import pytest

from mw_mcp_server.embeddings.queue import EmbeddingQueue, EmbeddingJob


def _make_job(title: str) -> EmbeddingJob:
    return EmbeddingJob(
        wiki_id="test",
        title=title,
        content="test content",
        namespace=0,
        last_modified=None,
    )


@pytest.mark.asyncio
async def test_enqueue_within_limit():
    """Jobs within max size should enqueue normally."""
    q = EmbeddingQueue(maxsize=5)
    size = await q.enqueue(_make_job("Page1"))
    assert size == 1


@pytest.mark.asyncio
async def test_enqueue_evicts_oldest_when_full():
    """When queue is full, oldest job should be evicted."""
    q = EmbeddingQueue(maxsize=2)
    await q.enqueue(_make_job("Page1"))
    await q.enqueue(_make_job("Page2"))

    # Queue is now full. Adding a third should evict Page1.
    size = await q.enqueue(_make_job("Page3"))
    assert size == 2

    # Verify Page1 was evicted — next job should be Page2
    job = await q.get_next_job()
    assert job.title == "Page2"


@pytest.mark.asyncio
async def test_get_next_job_returns_fifo():
    """Jobs should be returned in FIFO order."""
    q = EmbeddingQueue(maxsize=10)
    await q.enqueue(_make_job("A"))
    await q.enqueue(_make_job("B"))
    await q.enqueue(_make_job("C"))

    first = await q.get_next_job()
    assert first.title == "A"
    second = await q.get_next_job()
    assert second.title == "B"
