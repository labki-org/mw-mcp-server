"""
Session Cleanup

Provides functions for deleting expired chat sessions based on the
configured retention policy.

``run_periodic_cleanup()`` is a long-running coroutine meant to be started as
a background task from the application lifespan, e.g.::

    cleanup_task = asyncio.create_task(run_periodic_cleanup())
    ...
    cleanup_task.cancel()
"""

import asyncio
import logging
from datetime import timedelta
from typing import Callable, Optional

from sqlalchemy import delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from .models import ChatSession
from ..config import settings

logger = logging.getLogger("mcp.cleanup")


async def delete_expired_sessions(session: AsyncSession) -> int:
    """
    Delete chat sessions older than the configured retention period.

    The cutoff is computed in SQL (``now() - interval``) so it is compared in
    the same clock/timezone as the ``updated_at`` server default; comparing a
    timezone-aware Python datetime against the naive column fails in asyncpg.

    Parameters
    ----------
    session : AsyncSession
        SQLAlchemy async session.

    Returns
    -------
    int
        Number of sessions deleted.
    """
    retention = timedelta(days=settings.session_retention_days)

    result = await session.execute(
        delete(ChatSession).where(ChatSession.updated_at < func.now() - retention)
    )
    await session.commit()

    count = result.rowcount
    logger.info(
        "Deleted %d expired sessions (older than %d days)",
        count,
        settings.session_retention_days,
    )
    return count


async def run_periodic_cleanup(
    interval_seconds: Optional[float] = None,
    session_factory: Optional[Callable[[], AsyncSession]] = None,
) -> None:
    """
    Run ``delete_expired_sessions`` now and then every *interval_seconds*
    (default: ``session_cleanup_interval_seconds`` setting, else 6 hours)
    until cancelled. Failures are logged and retried on the next tick.
    """
    if session_factory is None:
        from .session import AsyncSessionLocal
        session_factory = AsyncSessionLocal
    interval = (
        interval_seconds if interval_seconds is not None
        else float(getattr(settings, "session_cleanup_interval_seconds", 6 * 3600))
    )

    while True:
        try:
            async with session_factory() as session:
                await delete_expired_sessions(session)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Expired session cleanup failed")
        await asyncio.sleep(interval)
