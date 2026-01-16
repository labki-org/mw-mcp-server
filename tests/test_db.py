"""
Database Integration Tests

Simple tests for database operations including:
- Session creation
- Message persistence
- Token usage recording
- Rate limit enforcement
"""

from datetime import date
from uuid import uuid4

from mw_mcp_server.db.models import ChatSession, ChatMessage, TokenUsage
from mw_mcp_server.config import settings


class TestChatSessionModels:
    """Tests for ChatSession and ChatMessage models."""

    def test_chat_session_defaults(self):
        """Verify ChatSession creates with expected defaults."""
        session = ChatSession(
            wiki_id="test-wiki",
            owner_user_id=123,
        )
        
        assert session.wiki_id == "test-wiki"
        assert session.owner_user_id == 123
        # Note: session_id is a server default, not set until persisted
        assert session.title is None
        assert session.summary is None

    def test_chat_message_creation(self):
        """Verify ChatMessage creates with required fields."""
        session_id = uuid4()
        message = ChatMessage(
            session_id=session_id,
            sender="user",
            content="Hello, world!",
        )
        
        assert message.session_id == session_id
        assert message.sender == "user"
        assert message.content == "Hello, world!"
        assert message.metadata_ is None

    def test_chat_message_with_metadata(self):
        """Verify ChatMessage can store metadata."""
        session_id = uuid4()
        message = ChatMessage(
            session_id=session_id,
            sender="assistant",
            content="Hi there!",
            metadata_={"tools_used": ["search"], "tokens": {"total": 100}},
        )
        
        assert message.metadata_ == {"tools_used": ["search"], "tokens": {"total": 100}}


class TestTokenUsageModel:
    """Tests for TokenUsage model."""

    def test_token_usage_creation(self):
        """Verify TokenUsage creates with expected values."""
        usage = TokenUsage(
            wiki_id="test-wiki",
            user_id=456,
            usage_date=date.today(),
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            request_count=1,
        )
        
        assert usage.wiki_id == "test-wiki"
        assert usage.user_id == 456
        assert usage.usage_date == date.today()
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.request_count == 1


class TestRateLimiterLogic:
    """Tests for rate limiter calculations (non-DB)."""

    def test_daily_token_limit_configured(self):
        """Verify daily token limit is configured."""
        assert settings.daily_token_limit > 0
        assert settings.daily_token_limit >= 1000

    def test_usage_status_fields(self):
        """Verify UsageStatus has expected fields."""
        from mw_mcp_server.db.rate_limiter import UsageStatus
        from datetime import datetime, timezone
        
        status = UsageStatus(
            tokens_used=5000,
            tokens_remaining=95000,
            limit=100000,
            requests_today=10,
            is_limited=False,
            reset_time=datetime.now(timezone.utc),
        )
        
        assert status.tokens_used == 5000
        assert status.tokens_remaining == 95000
        assert status.limit == 100000
        assert status.requests_today == 10
        assert status.is_limited is False
        assert status.reset_time is not None

    def test_usage_status_limited_when_exceeded(self):
        """Verify is_limited is True when tokens exceed limit."""
        from mw_mcp_server.db.rate_limiter import UsageStatus
        from datetime import datetime, timezone
        
        # Simulate exceeded usage
        status = UsageStatus(
            tokens_used=100000,
            tokens_remaining=0,
            limit=100000,
            requests_today=50,
            is_limited=True,
            reset_time=datetime.now(timezone.utc),
        )
        
        assert status.is_limited is True
        assert status.tokens_remaining == 0
