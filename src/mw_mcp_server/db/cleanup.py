"""
Session Cleanup

Provides functions for deleting expired chat sessions based on the
configured retention policy.

Can be invoked as a scheduled task (e.g., cron or Celery beat).
"""

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from .models import ChatSession
from ..config import settings

logger = logging.getLogger("mcp.cleanup")


async def delete_expired_sessions(session: AsyncSession) -> int:
    """
    Delete chat sessions older than the configured retention period.

    Parameters
    ----------
    session : AsyncSession
        SQLAlchemy async session.

    Returns
    -------
    int
        Number of sessions deleted.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=settings.session_retention_days)

    result = await session.execute(
        delete(ChatSession).where(ChatSession.updated_at < cutoff)
    )
    await session.commit()

    count = result.rowcount
    logger.info(
        "Deleted %d expired sessions (older than %d days, cutoff=%s)",
        count,
        settings.session_retention_days,
        cutoff.isoformat(),
    )
    return count
