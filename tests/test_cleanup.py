"""
Session cleanup tests (need PostgreSQL; skipped when unavailable).
"""

import asyncio
import uuid

from sqlalchemy import select, text

from mw_mcp_server.config import settings
from mw_mcp_server.db.cleanup import delete_expired_sessions, run_periodic_cleanup
from mw_mcp_server.db.models import ChatSession

from pg_utils import pg_session_factory_fixture  # noqa: F401  (registers fixture)


async def _make_sessions(sf):
    old_id, new_id = uuid.uuid4(), uuid.uuid4()
    async with sf() as s:
        s.add(ChatSession(session_id=old_id, wiki_id="w", owner_user_id=1))
        s.add(ChatSession(session_id=new_id, wiki_id="w", owner_user_id=1))
        await s.flush()
        await s.execute(
            text(
                "UPDATE chat_session SET updated_at = now() - make_interval(days => :d) "
                "WHERE session_id = :id"
            ),
            {"d": settings.session_retention_days + 1, "id": old_id},
        )
        await s.commit()
    return old_id, new_id


async def _remaining(sf):
    async with sf() as s:
        return set((await s.execute(select(ChatSession.session_id))).scalars())


async def test_delete_expired_sessions(pg_session_factory):
    old_id, new_id = await _make_sessions(pg_session_factory)
    async with pg_session_factory() as s:
        assert await delete_expired_sessions(s) == 1
    assert await _remaining(pg_session_factory) == {new_id}


async def test_periodic_cleanup_runs_immediately(pg_session_factory):
    _, new_id = await _make_sessions(pg_session_factory)
    task = asyncio.create_task(run_periodic_cleanup(3600, pg_session_factory))
    try:
        for _ in range(50):
            if await _remaining(pg_session_factory) == {new_id}:
                break
            await asyncio.sleep(0.05)
        assert await _remaining(pg_session_factory) == {new_id}
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
