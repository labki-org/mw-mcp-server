"""
Tool Dispatch Tests

Tests for the tool dispatch layer's input validation and routing.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from mw_mcp_server.tools.base import dispatch_tool_call, MAX_TOOL_STRING_ARG_LENGTH


class TestToolDispatch:
    """Tests for dispatch_tool_call."""

    @pytest.mark.asyncio
    async def test_unknown_tool_raises_value_error(self):
        """Unknown tool name should raise ValueError."""
        user = MagicMock()
        vs = MagicMock()
        embedder = MagicMock()

        with pytest.raises(ValueError, match="Unknown tool"):
            await dispatch_tool_call("nonexistent_tool", {}, user, vs, embedder)

    @pytest.mark.asyncio
    async def test_oversized_string_arg_rejected(self):
        """String arguments exceeding max length should be rejected."""
        user = MagicMock()
        vs = MagicMock()
        embedder = MagicMock()

        oversized_args = {"title": "x" * (MAX_TOOL_STRING_ARG_LENGTH + 1)}

        with pytest.raises(ValueError, match="exceeds maximum length"):
            await dispatch_tool_call("mw_get_page", oversized_args, user, vs, embedder)

    @pytest.mark.asyncio
    async def test_normal_length_arg_accepted(self):
        """Normal-length string arguments should pass validation (tool may still fail)."""
        user = MagicMock()
        user.wiki_id = "test"
        user.allowed_namespaces = [0]
        vs = MagicMock()
        embedder = MagicMock()

        # The actual tool handler will fail because we're using mocks,
        # but the arg-length validation should pass
        try:
            await dispatch_tool_call("mw_get_page", {"title": "Main Page"}, user, vs, embedder)
        except ValueError as e:
            # Should NOT be the "exceeds maximum length" error
            assert "exceeds maximum length" not in str(e)
        except Exception:
            # Other errors are fine — we're testing the validation layer, not the tool
            pass
