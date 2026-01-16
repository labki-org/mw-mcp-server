"""
Rate Limiter

Provides token-based rate limiting using daily token quotas.
Tracks usage in PostgreSQL and enforces limits per user per wiki.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import NamedTuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .models import TokenUsage
from ..config import settings


class UsageStatus(NamedTuple):
    """Status of a user's token usage for the current day."""
    tokens_used: int
    tokens_remaining: int
    limit: int
    requests_today: int
    is_limited: bool
    reset_time: datetime


class RateLimiter:
    """
    Token-based rate limiter using PostgreSQL for persistence.
    
    Tracks daily token usage per user per wiki and enforces
    configurable daily limits.
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
        self._daily_limit = settings.daily_token_limit

    async def check_limit(
        self,
        wiki_id: str,
        user_id: int,
    ) -> UsageStatus:
        """
        Check if a user is within their daily token limit.
        
        Parameters
        ----------
        wiki_id : str
            Tenant wiki identifier.
        user_id : int
            MediaWiki user ID.
            
        Returns
        -------
        UsageStatus
            Current usage status including remaining tokens.
        """
        today = date.today()
        
        result = await self._session.execute(
            select(TokenUsage).where(
                TokenUsage.wiki_id == wiki_id,
                TokenUsage.user_id == user_id,
                TokenUsage.usage_date == today,
            )
        )
        usage = result.scalar_one_or_none()
        
        tokens_used = usage.total_tokens if usage else 0
        requests_today = usage.request_count if usage else 0
        tokens_remaining = max(0, self._daily_limit - tokens_used)
        is_limited = tokens_used >= self._daily_limit
        
        # Reset time is midnight UTC of the next day
        tomorrow = datetime.combine(
            today + timedelta(days=1),
            datetime.min.time(),
            tzinfo=timezone.utc,
        )
        
        return UsageStatus(
            tokens_used=tokens_used,
            tokens_remaining=tokens_remaining,
            limit=self._daily_limit,
            requests_today=requests_today,
            is_limited=is_limited,
            reset_time=tomorrow,
        )

    async def record_usage(
        self,
        wiki_id: str,
        user_id: int,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> UsageStatus:
        """
        Record token usage for an LLM request.
        
        Uses PostgreSQL upsert to atomically update the daily record.
        
        Parameters
        ----------
        wiki_id : str
            Tenant wiki identifier.
        user_id : int
            MediaWiki user ID.
        prompt_tokens : int
            Number of prompt tokens used.
        completion_tokens : int
            Number of completion tokens used.
            
        Returns
        -------
        UsageStatus
            Updated usage status after recording.
        """
        today = date.today()
        total_tokens = prompt_tokens + completion_tokens
        
        # Upsert: insert or update on conflict
        stmt = pg_insert(TokenUsage).values(
            wiki_id=wiki_id,
            user_id=user_id,
            usage_date=today,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            request_count=1,
        ).on_conflict_do_update(
            constraint="uq_usage_user_date",
            set_={
                "prompt_tokens": TokenUsage.prompt_tokens + prompt_tokens,
                "completion_tokens": TokenUsage.completion_tokens + completion_tokens,
                "total_tokens": TokenUsage.total_tokens + total_tokens,
                "request_count": TokenUsage.request_count + 1,
            },
        )
        
        await self._session.execute(stmt)
        await self._session.flush()
        
        return await self.check_limit(wiki_id, user_id)

    async def get_usage_history(
        self,
        wiki_id: str,
        user_id: int,
        days: int = 7,
    ) -> list[dict]:
        """
        Get recent token usage history for a user.
        
        Parameters
        ----------
        wiki_id : str
            Tenant wiki identifier.
        user_id : int
            MediaWiki user ID.
        days : int
            Number of days of history to return.
            
        Returns
        -------
        list[dict]
            List of daily usage records.
        """
        start_date = date.today() - timedelta(days=days)
        
        result = await self._session.execute(
            select(TokenUsage)
            .where(
                TokenUsage.wiki_id == wiki_id,
                TokenUsage.user_id == user_id,
                TokenUsage.usage_date >= start_date,
            )
            .order_by(TokenUsage.usage_date.desc())
        )
        
        return [
            {
                "date": usage.usage_date.isoformat(),
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "request_count": usage.request_count,
            }
            for usage in result.scalars().all()
        ]
