"""
Rate Limiter

Provides token-based rate limiting using daily token quotas.
Tracks usage in PostgreSQL and enforces limits per user per wiki.

Chat requests use ``reserve()`` / ``reconcile()`` rather than a separate
check-then-record: the reservation is an atomic conditional upsert, so N
parallel requests can't all pass a stale pre-check and overshoot the quota.
Days are UTC days, matching the advertised UTC-midnight reset.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Callable, NamedTuple, Optional

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .models import TokenUsage
from ..config import settings


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


def _reset_time(day: date) -> datetime:
    """Midnight UTC at the end of *day*."""
    return datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)


class UsageStatus(NamedTuple):
    """Status of a user's token usage for the current day."""
    tokens_used: int
    tokens_remaining: int
    limit: int
    requests_today: int
    is_limited: bool
    reset_time: datetime


class Reservation(NamedTuple):
    """Tokens provisionally charged to a user's daily budget."""
    wiki_id: str
    user_id: int
    usage_date: date
    reserved_tokens: int


class RateLimiter:
    """
    Token-based rate limiter using PostgreSQL for persistence.
    
    Tracks daily token usage per user per wiki and enforces
    configurable daily limits.
    """

    def __init__(
        self,
        session: AsyncSession,
        session_factory: Optional[Callable[[], AsyncSession]] = None,
        daily_limit: Optional[int] = None,
    ) -> None:
        """
        Initialize with an async database session.

        Parameters
        ----------
        session : AsyncSession
            SQLAlchemy async session for read-only queries.
        session_factory : Optional[Callable[[], AsyncSession]]
            Factory for the short, independently committed transactions used
            by ``reserve``/``reconcile``. Defaults to ``AsyncSessionLocal``.
            Using a separate session means the row lock is held only for the
            upsert itself (not for the whole LLM request), and the charge is
            durable even if the request's own transaction is rolled back.
        daily_limit : Optional[int]
            Override for ``settings.daily_token_limit``.
        """
        self._session = session
        self._session_factory = session_factory
        self._daily_limit = daily_limit if daily_limit is not None else settings.daily_token_limit

    def _independent_session(self) -> AsyncSession:
        if self._session_factory is None:
            from .session import AsyncSessionLocal
            self._session_factory = AsyncSessionLocal
        return self._session_factory()

    def _status(self, tokens_used: int, requests_today: int, day: date) -> UsageStatus:
        return UsageStatus(
            tokens_used=tokens_used,
            tokens_remaining=max(0, self._daily_limit - tokens_used),
            limit=self._daily_limit,
            requests_today=requests_today,
            is_limited=tokens_used >= self._daily_limit,
            reset_time=_reset_time(day),
        )

    async def reserve(
        self,
        wiki_id: str,
        user_id: int,
        tokens: int,
    ) -> tuple[UsageStatus, Optional[Reservation]]:
        """
        Atomically admit a request and provisionally charge *tokens*.

        The charge is applied only while the user's running total is still
        under the daily limit, in a single ``INSERT ... ON CONFLICT DO UPDATE
        ... WHERE`` statement, so concurrent requests serialise on the row and
        each sees the others' reservations. Committed immediately.

        Returns ``(status, reservation)``; ``reservation`` is None when the
        user is over the limit (``status.is_limited`` is then True). Pass the
        reservation to ``reconcile`` once actual usage is known.
        """
        today = _utc_today()
        tokens = max(0, int(tokens))

        stmt = (
            pg_insert(TokenUsage)
            .values(
                wiki_id=wiki_id,
                user_id=user_id,
                usage_date=today,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=tokens,
                request_count=1,
            )
            .on_conflict_do_update(
                constraint="uq_usage_user_date",
                set_={
                    "total_tokens": TokenUsage.total_tokens + tokens,
                    "request_count": TokenUsage.request_count + 1,
                },
                where=TokenUsage.total_tokens < self._daily_limit,
            )
            .returning(TokenUsage.total_tokens, TokenUsage.request_count)
        )

        async with self._independent_session() as session:
            row = (await session.execute(stmt)).one_or_none()
            if row is None:
                # Conflict row exists and is at/over the limit: not admitted.
                current = (
                    await session.execute(
                        select(TokenUsage.total_tokens, TokenUsage.request_count).where(
                            TokenUsage.wiki_id == wiki_id,
                            TokenUsage.user_id == user_id,
                            TokenUsage.usage_date == today,
                        )
                    )
                ).one()
                await session.rollback()
                status = self._status(current[0], current[1], today)
                # Report limited even if a concurrent reconcile just freed budget.
                return status._replace(is_limited=True), None
            await session.commit()

        tokens_used, requests_today = row
        reservation = Reservation(wiki_id, user_id, today, tokens)
        # Admitted, so not limited for *this* request even if the reservation
        # pushed the running total past the limit.
        return self._status(tokens_used, requests_today, today)._replace(is_limited=False), reservation

    async def reconcile(
        self,
        reservation: Reservation,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """
        Replace a reservation with the actual usage. Committed immediately.

        Applies to the reservation's day even if the request crossed midnight,
        so the provisional charge is always released from the same row.
        """
        actual = prompt_tokens + completion_tokens
        delta = actual - reservation.reserved_tokens
        async with self._independent_session() as session:
            await session.execute(
                update(TokenUsage)
                .where(
                    TokenUsage.wiki_id == reservation.wiki_id,
                    TokenUsage.user_id == reservation.user_id,
                    TokenUsage.usage_date == reservation.usage_date,
                )
                .values(
                    prompt_tokens=TokenUsage.prompt_tokens + prompt_tokens,
                    completion_tokens=TokenUsage.completion_tokens + completion_tokens,
                    total_tokens=func.greatest(0, TokenUsage.total_tokens + delta),
                )
            )
            await session.commit()

    async def check_limit(
        self,
        wiki_id: str,
        user_id: int,
    ) -> UsageStatus:
        """
        Check if a user is within their daily token limit (read-only; use
        ``reserve`` to admit a request).

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
        today = _utc_today()

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
        return self._status(tokens_used, requests_today, today)

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
        today = _utc_today()
        total_tokens = prompt_tokens + completion_tokens

        stmt = (
            pg_insert(TokenUsage)
            .values(
                wiki_id=wiki_id,
                user_id=user_id,
                usage_date=today,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                request_count=1,
            )
            .on_conflict_do_update(
                constraint="uq_usage_user_date",
                set_={
                    "prompt_tokens": TokenUsage.prompt_tokens + prompt_tokens,
                    "completion_tokens": TokenUsage.completion_tokens + completion_tokens,
                    "total_tokens": TokenUsage.total_tokens + total_tokens,
                    "request_count": TokenUsage.request_count + 1,
                },
            )
            .returning(TokenUsage.total_tokens, TokenUsage.request_count)
        )

        result = await self._session.execute(stmt)
        tokens_used, requests_today = result.one()
        return self._status(tokens_used, requests_today, today)

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
        start_date = _utc_today() - timedelta(days=days)
        
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
