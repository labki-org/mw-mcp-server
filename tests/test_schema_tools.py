"""
Tests for tools/schema_tools.py — particularly the semantic-suggestions
fallback that catches cases like 'Lab member' → Person/Researcher.
"""

from unittest.mock import AsyncMock

import pytest

from mw_mcp_server.tools.schema_tools import (
    tool_get_categories,
    tool_get_properties,
    NS_CATEGORY,
    NS_PROPERTY,
)


def _make_vs(pages: list, search_results: list = ()):
    """Build a VectorStore double with the two methods schema_tools touches."""
    vs = AsyncMock()
    vs.get_pages_by_namespace.return_value = list(pages)
    vs.search.return_value = list(search_results)
    return vs


def _make_embedder(vector=None):
    emb = AsyncMock()
    emb.embed.return_value = [vector or [0.1, 0.2, 0.3]]
    return emb


# ---------------------------------------------------------------------
# names mode
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_categories_names_mode_returns_found_and_missing():
    """Names mode should partition input into found / missing buckets."""
    vs = _make_vs(["Category:Person", "Category:Researcher"])
    emb = _make_embedder()

    result = await tool_get_categories(
        vector_store=vs,
        wiki_id="w",
        names=["Person", "LabMember"],
        embedder=emb,
        allowed_namespaces=[NS_CATEGORY],
    )

    assert result["found"] == ["Category:Person"]
    assert result["missing"] == ["LabMember"]


@pytest.mark.asyncio
async def test_categories_names_mode_emits_suggestions_for_missing():
    """When a name doesn't exist, the user-asked concept should drive
    a vector search that surfaces related categories."""
    vs = _make_vs(
        pages=["Category:Person", "Category:Researcher"],
        search_results=[
            ("Category:Researcher", None, NS_CATEGORY, 0.9),
            ("Category:Person", None, NS_CATEGORY, 0.85),  # already found, excluded
            ("Category:Organizational role", None, NS_CATEGORY, 0.7),
        ],
    )
    emb = _make_embedder()

    result = await tool_get_categories(
        vector_store=vs,
        wiki_id="w",
        names=["Person", "LabMember"],
        embedder=emb,
        allowed_namespaces=[NS_CATEGORY],
    )

    # Person was found, so it should NOT appear in suggestions.
    assert "Category:Person" not in result["suggestions"]
    # Researcher and Org role are conceptually adjacent — they should.
    assert "Category:Researcher" in result["suggestions"]
    assert "Category:Organizational role" in result["suggestions"]
    # Vector search should have used the missing name as the query.
    emb.embed.assert_awaited_once()
    assert "LabMember" in emb.embed.call_args.args[0][0]


@pytest.mark.asyncio
async def test_categories_names_mode_no_suggestions_when_all_found():
    """If every requested name exists, suggestions are unnecessary — and
    we shouldn't waste an embedding round-trip."""
    vs = _make_vs(["Category:Person"])
    emb = _make_embedder()

    result = await tool_get_categories(
        vector_store=vs,
        wiki_id="w",
        names=["Person"],
        embedder=emb,
        allowed_namespaces=[NS_CATEGORY],
    )

    assert result["found"] == ["Category:Person"]
    assert result["missing"] == []
    assert result["suggestions"] == []
    emb.embed.assert_not_awaited()


# ---------------------------------------------------------------------
# prefix / list mode
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_categories_prefix_with_few_matches_adds_suggestions():
    """A prefix that matches only one or zero pages should pull in
    semantic alternatives so the LLM has something to investigate."""
    vs = _make_vs(
        pages=["Category:Lab"],  # the lone direct hit
        search_results=[
            ("Category:Lab", None, NS_CATEGORY, 0.95),  # already in matches
            ("Category:Researcher", None, NS_CATEGORY, 0.8),
            ("Category:Person", None, NS_CATEGORY, 0.7),
        ],
    )
    emb = _make_embedder()

    result = await tool_get_categories(
        vector_store=vs,
        wiki_id="w",
        prefix="Lab",
        embedder=emb,
        allowed_namespaces=[NS_CATEGORY],
    )

    assert result["matches"] == ["Category:Lab"]
    assert "Category:Lab" not in result["suggestions"]
    assert "Category:Researcher" in result["suggestions"]


@pytest.mark.asyncio
async def test_categories_prefix_with_many_matches_skips_suggestions():
    """When prefix matches plenty of pages, save the embedding cost."""
    vs = _make_vs(
        pages=["Category:A", "Category:B", "Category:C", "Category:D"],
    )
    emb = _make_embedder()

    result = await tool_get_categories(
        vector_store=vs,
        wiki_id="w",
        prefix="C",
        embedder=emb,
        allowed_namespaces=[NS_CATEGORY],
    )

    assert len(result["matches"]) >= 3
    assert result["suggestions"] == []
    emb.embed.assert_not_awaited()


@pytest.mark.asyncio
async def test_categories_no_prefix_skips_suggestions():
    """Bare 'list everything' shouldn't trigger the semantic path —
    we have no concept to embed."""
    vs = _make_vs(pages=["Category:A"])
    emb = _make_embedder()

    result = await tool_get_categories(
        vector_store=vs,
        wiki_id="w",
        embedder=emb,
        allowed_namespaces=[NS_CATEGORY],
    )

    assert result["suggestions"] == []
    emb.embed.assert_not_awaited()


# ---------------------------------------------------------------------
# permission gate
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_categories_namespace_locked_out():
    """Users without Category-namespace access should get a noted empty result."""
    vs = _make_vs(pages=["Category:Person"])
    emb = _make_embedder()

    result = await tool_get_categories(
        vector_store=vs,
        wiki_id="w",
        prefix="Person",
        embedder=emb,
        allowed_namespaces=[0],  # main namespace only
    )

    assert result["matches"] == []
    assert result["suggestions"] == []
    assert "not accessible" in result["note"]
    # No DB or embedder traffic when access is denied up front.
    vs.get_pages_by_namespace.assert_not_awaited()
    emb.embed.assert_not_awaited()


# ---------------------------------------------------------------------
# properties — same backbone, smoke test only
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_properties_uses_property_namespace_and_prefix():
    vs = _make_vs(
        pages=["Property:Has center role"],
        search_results=[
            ("Property:Has affiliation", None, NS_PROPERTY, 0.85),
        ],
    )
    emb = _make_embedder()

    result = await tool_get_properties(
        vector_store=vs,
        wiki_id="w",
        names=["affiliation"],
        embedder=emb,
        allowed_namespaces=[NS_PROPERTY],
    )

    assert result["found"] == []
    assert result["missing"] == ["affiliation"]
    assert "Property:Has affiliation" in result["suggestions"]


# ---------------------------------------------------------------------
# embedder optional
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_categories_works_without_embedder():
    """The tool should degrade gracefully when no embedder is wired —
    matches still come back, suggestions are simply empty."""
    vs = _make_vs(pages=["Category:Person"])

    result = await tool_get_categories(
        vector_store=vs,
        wiki_id="w",
        names=["Person", "LabMember"],
        embedder=None,
        allowed_namespaces=[NS_CATEGORY],
    )

    assert result["found"] == ["Category:Person"]
    assert result["missing"] == ["LabMember"]
    assert result["suggestions"] == []
