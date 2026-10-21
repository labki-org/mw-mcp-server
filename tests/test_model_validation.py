"""
Model Validation Tests

Tests for Pydantic request model constraints including
max_length, max_items, and role validation.
"""

import pytest
from pydantic import ValidationError

from mw_mcp_server.api.models import (
    ChatMessage,
    ChatRequest,
    SearchRequest,
    SMWQueryRequest,
)


class TestChatMessageValidation:
    """Tests for ChatMessage field constraints."""

    def test_valid_message(self):
        msg = ChatMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_empty_content_rejected(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content="")

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="hacker", content="hello")

    def test_content_max_length_enforced(self):
        """Content exceeding 100k chars should be rejected."""
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content="x" * 100_001)

    def test_content_at_max_length_accepted(self):
        msg = ChatMessage(role="user", content="x" * 100_000)
        assert len(msg.content) == 100_000


class TestChatRequestValidation:
    """Tests for ChatRequest field constraints."""

    def test_valid_request(self):
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="hello")],
        )
        assert len(req.messages) == 1

    def test_empty_messages_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(messages=[])

    def test_max_messages_enforced(self):
        """More than 100 messages should be rejected."""
        msgs = [ChatMessage(role="user", content="hello")] * 101
        with pytest.raises(ValidationError):
            ChatRequest(messages=msgs)

    def test_100_messages_accepted(self):
        msgs = [ChatMessage(role="user", content="hello")] * 100
        req = ChatRequest(messages=msgs)
        assert len(req.messages) == 100

    def test_session_id_max_length_enforced(self):
        with pytest.raises(ValidationError):
            ChatRequest(
                messages=[ChatMessage(role="user", content="hello")],
                session_id="x" * 65,
            )

    def test_invalid_context_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(
                messages=[ChatMessage(role="user", content="hello")],
                context="invalid",
            )


class TestSearchRequestValidation:
    """Tests for SearchRequest constraints."""

    def test_query_max_length_enforced(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="x" * 10_001)

    def test_valid_search(self):
        req = SearchRequest(query="find me something")
        assert req.query == "find me something"


class TestSMWQueryRequestValidation:
    """Tests for SMWQueryRequest constraints."""

    def test_ask_max_length_enforced(self):
        with pytest.raises(ValidationError):
            SMWQueryRequest(ask="x" * 10_001)

    def test_valid_smw_query(self):
        req = SMWQueryRequest(ask="[[Category:Test]]|?Property")
        assert "Category:Test" in req.ask


class TestClientRoleAndSizeBounds:
    """Client-supplied roles and sizes are bounded."""

    @pytest.mark.parametrize("role", ["tool", "function", "developer"])
    def test_privileged_roles_rejected(self, role):
        with pytest.raises(ValidationError):
            ChatMessage(role=role, content="hello")

    def test_system_role_downgraded_to_user(self):
        """MWAssistant sends editor context as a 'system' message."""
        msg = ChatMessage(role="system", content="Context:\nfoo")
        assert msg.role == "user"

    def test_total_request_chars_capped(self):
        from mw_mcp_server.api.models import MAX_CHAT_REQUEST_CHARS

        per = 100_000
        n = MAX_CHAT_REQUEST_CHARS // per + 1
        with pytest.raises(ValidationError, match="Total message content"):
            ChatRequest(messages=[ChatMessage(role="user", content="x" * per)] * n)

    def test_total_request_at_cap_accepted(self):
        req = ChatRequest(messages=[ChatMessage(role="user", content="x" * 100_000)] * 2)
        assert len(req.messages) == 2

    def test_embedding_content_capped(self):
        from mw_mcp_server.api.models import (
            EmbeddingUpdatePageRequest,
            MAX_EMBEDDING_CONTENT_CHARS,
        )

        EmbeddingUpdatePageRequest(title="T", content="x" * MAX_EMBEDDING_CONTENT_CHARS)
        with pytest.raises(ValidationError):
            EmbeddingUpdatePageRequest(title="T", content="x" * (MAX_EMBEDDING_CONTENT_CHARS + 1))

    def test_embedding_title_capped(self):
        from mw_mcp_server.api.models import EmbeddingDeletePageRequest

        with pytest.raises(ValidationError):
            EmbeddingDeletePageRequest(title="x" * 513)
