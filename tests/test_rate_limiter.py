"""
Rate Limiter Tests

Tests for token-based rate limiting logic.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from mw_mcp_server.db.rate_limiter import RateLimiter


class FakeTokenUsage:
    """Minimal fake for TokenUsage rows returned by queries."""
    def __init__(self, total_tokens=0, request_count=0):
        self.total_tokens = total_tokens
        self.request_count = request_count


class TestRateLimiterCheckLimit:
    """Tests for RateLimiter.check_limit."""

    @pytest.mark.asyncio
    async def test_no_prior_usage_returns_unlimited(self):
        """User with no usage today should not be rate limited."""
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=result_mock)

        with patch("mw_mcp_server.db.rate_limiter.settings") as mock_settings:
            mock_settings.daily_token_limit = 100_000
            limiter = RateLimiter(session)
            status = await limiter.check_limit("wiki1", 42)

        assert status.is_limited is False
        assert status.tokens_used == 0
        assert status.tokens_remaining == 100_000

    @pytest.mark.asyncio
    async def test_at_limit_returns_limited(self):
        """User who hit the limit should be rate limited."""
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = FakeTokenUsage(
            total_tokens=100_000, request_count=50
        )
        session.execute = AsyncMock(return_value=result_mock)

        with patch("mw_mcp_server.db.rate_limiter.settings") as mock_settings:
            mock_settings.daily_token_limit = 100_000
            limiter = RateLimiter(session)
            status = await limiter.check_limit("wiki1", 42)

        assert status.is_limited is True
        assert status.tokens_remaining == 0

    @pytest.mark.asyncio
    async def test_under_limit_returns_unlimited(self):
        """User under the limit should not be rate limited."""
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = FakeTokenUsage(
            total_tokens=50_000, request_count=10
        )
        session.execute = AsyncMock(return_value=result_mock)

        with patch("mw_mcp_server.db.rate_limiter.settings") as mock_settings:
            mock_settings.daily_token_limit = 100_000
            limiter = RateLimiter(session)
            status = await limiter.check_limit("wiki1", 42)

        assert status.is_limited is False
        assert status.tokens_remaining == 50_000
        assert status.requests_today == 10

    @pytest.mark.asyncio
    async def test_reset_time_is_next_midnight_utc(self):
        """Reset time should be midnight UTC of the next day."""
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=result_mock)

        with patch("mw_mcp_server.db.rate_limiter.settings") as mock_settings:
            mock_settings.daily_token_limit = 100_000
            limiter = RateLimiter(session)
            status = await limiter.check_limit("wiki1", 42)

        today = datetime.now(timezone.utc).date()  # quota days are UTC days
        expected_reset = datetime.combine(
            today + timedelta(days=1),
            datetime.min.time(),
            tzinfo=timezone.utc,
        )
        assert status.reset_time == expected_reset


# ---------------------------------------------------------------------
# Atomic reservation (needs PostgreSQL; skipped when unavailable)
# ---------------------------------------------------------------------

from pg_utils import pg_session_factory_fixture  # noqa: E402,F401  (registers fixture)


def _limiter(sf, limit):
    return RateLimiter(session=None, session_factory=sf, daily_limit=limit)


class TestRateLimiterReserve:
    """Concurrency-safe reserve/reconcile against a real database."""

    @pytest.mark.asyncio
    async def test_parallel_requests_cannot_all_pass(self, pg_session_factory):
        """N parallel reservations must not all be admitted on a stale total."""
        import asyncio

        results = await asyncio.gather(
            *[
                _limiter(pg_session_factory, 10_000).reserve("wiki1", 42, 4_000)
                for _ in range(8)
            ]
        )
        admitted = [r for _, r in results if r is not None]
        # 0 -> 4k -> 8k -> 12k: the 4th onward sees total >= limit.
        assert len(admitted) == 3
        assert all(status.is_limited for status, r in results if r is None)

    @pytest.mark.asyncio
    async def test_reconcile_replaces_reservation_with_actual(self, pg_session_factory):
        limiter = _limiter(pg_session_factory, 10_000)
        status, reservation = await limiter.reserve("wiki1", 42, 4_000)
        assert reservation is not None and status.tokens_used == 4_000

        await limiter.reconcile(reservation, prompt_tokens=700, completion_tokens=300)

        async with pg_session_factory() as s:
            limiter = RateLimiter(s, session_factory=pg_session_factory, daily_limit=10_000)
            status = await limiter.check_limit("wiki1", 42)
        assert status.tokens_used == 1_000
        assert status.requests_today == 1

    @pytest.mark.asyncio
    async def test_over_limit_user_is_rejected(self, pg_session_factory):
        limiter = _limiter(pg_session_factory, 1_000)
        _, first = await limiter.reserve("wiki1", 42, 500)
        await limiter.reconcile(first, prompt_tokens=900, completion_tokens=300)

        status, reservation = await limiter.reserve("wiki1", 42, 500)
        assert reservation is None
        assert status.is_limited
        assert status.tokens_used == 1_200
