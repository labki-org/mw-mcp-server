"""
Rate Limiter Tests

Tests for token-based rate limiting logic.
"""

import pytest
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from mw_mcp_server.db.rate_limiter import RateLimiter, UsageStatus


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

        today = date.today()
        expected_reset = datetime.combine(
            today + timedelta(days=1),
            datetime.min.time(),
            tzinfo=timezone.utc,
        )
        assert status.reset_time == expected_reset
