"""
Tests for tools/pagination.py — the truncation-metadata wrapper used by
every listy LLM tool. The note text is part of the contract: it's what
nudges the LLM toward widening `limit` instead of treating a capped
result as the full set.
"""

import pytest

from mw_mcp_server.tools.pagination import paginated


def test_paginated_under_limit_not_truncated():
    out = paginated(["a", "b"], limit=10)
    assert out["results"] == ["a", "b"]
    assert out["count"] == 2
    assert out["limit"] == 10
    assert out["truncated"] is False
    assert "no truncation" in out["note"].lower()


def test_paginated_at_limit_flags_truncated():
    out = paginated(["a", "b", "c"], limit=3)
    assert out["truncated"] is True
    # The note must point the LLM at concrete next steps, not just announce
    # the cap. Failing this check means the LLM loses its actionable hint.
    note = out["note"].lower()
    assert "more" in note
    assert "limit" in note


def test_paginated_custom_label():
    out = paginated(["x"], limit=5, label="matches")
    assert "matches" in out and "results" not in out
    assert out["matches"] == ["x"]


def test_paginated_extra_fields_merged():
    out = paginated([], limit=5, label="members", extra={"category": "Foo"})
    assert out["category"] == "Foo"
    assert out["members"] == []


@pytest.mark.asyncio
async def test_list_pages_returns_paginated_envelope():
    """tool_list_pages must wrap its result so the LLM sees `truncated`."""
    from unittest.mock import AsyncMock

    from mw_mcp_server.tools.schema_tools import tool_list_pages

    vs = AsyncMock()
    # Return enough rows that limit (3) caps it.
    vs.get_pages_by_namespace.return_value = ["P1", "P2", "P3", "P4"]

    out = await tool_list_pages(
        vector_store=vs,
        wiki_id="w",
        namespace=0,
        limit=3,
        allowed_namespaces=[0],
    )
    assert out["results"] == ["P1", "P2", "P3"]
    assert out["truncated"] is True


@pytest.mark.asyncio
async def test_categories_prefix_mode_includes_truncation_metadata():
    """Existing prefix-mode `matches`/`suggestions` shape stays, plus
    truncation fields on top."""
    from unittest.mock import AsyncMock

    from mw_mcp_server.tools.schema_tools import tool_get_categories, NS_CATEGORY

    vs = AsyncMock()
    vs.get_pages_by_namespace.return_value = [
        f"Category:C{i}" for i in range(10)
    ]
    vs.search.return_value = []

    emb = AsyncMock()
    emb.embed.return_value = [[0.1, 0.2]]

    out = await tool_get_categories(
        vector_store=vs,
        wiki_id="w",
        prefix="C",
        limit=3,
        allowed_namespaces=[NS_CATEGORY],
        embedder=emb,
    )
    assert "matches" in out
    assert "suggestions" in out
    assert out["limit"] == 3
    assert out["truncated"] is True
