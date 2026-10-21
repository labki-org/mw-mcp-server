"""
Tool-layer permission filtering, argument bounds and re-embedding tests.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from mw_mcp_server.auth.models import UserContext
from mw_mcp_server.tools import base, wiki_tools
from mw_mcp_server.tools.access import filter_readable
from mw_mcp_server.tools.schema_tools import (
    NS_CATEGORY,
    tool_get_categories,
    tool_list_pages,
)
from mw_mcp_server.wiki.api_client import PageContent


@pytest.fixture
def user() -> UserContext:
    return UserContext(
        username="Alice", user_id=42, wiki_id="w", client_id="MWAssistant",
        allowed_namespaces=[0, NS_CATEGORY],
    )


def _access_client(hidden: set):
    return SimpleNamespace(check_read_access=AsyncMock(
        side_effect=lambda titles, user: {t: t not in hidden for t in titles}
    ))


# ---------------------------------------------------------------------
# filter_readable
# ---------------------------------------------------------------------

async def test_filter_readable_fills_limit_past_hidden_titles(user):
    titles = [f"P{i}" for i in range(250)]
    hidden = {f"P{i}" for i in range(150)}
    client = _access_client(hidden)

    readable, truncated = await filter_readable(titles, user, 10, client=client)

    assert readable == [f"P{i}" for i in range(150, 160)]
    assert truncated is True
    # Chunks never exceed the wiki's 100-title limit.
    assert all(len(c.args[0]) <= 100 for c in client.check_read_access.await_args_list)


async def test_filter_readable_exact_fit_not_truncated(user):
    readable, truncated = await filter_readable(["A", "B"], user, 2, client=_access_client(set()))
    assert readable == ["A", "B"]
    assert truncated is False


# ---------------------------------------------------------------------
# schema tools
# ---------------------------------------------------------------------

async def test_list_pages_hides_unreadable_titles(user):
    vs = AsyncMock()
    vs.get_pages_by_namespace.return_value = ["Public", "Secret", "Other"]

    out = await tool_list_pages(
        vector_store=vs, wiki_id="w", namespace=0, limit=50,
        allowed_namespaces=[0], user=user, access_client=_access_client({"Secret"}),
    )

    assert out["results"] == ["Public", "Other"]
    assert out["truncated"] is False


async def test_categories_names_mode_reports_unreadable_as_missing(user):
    vs = AsyncMock()
    vs.get_pages_by_namespace.return_value = ["Category:Public", "Category:Secret"]
    vs.search.return_value = [("Category:Secret", None, NS_CATEGORY, 0.9),
                              ("Category:Other", None, NS_CATEGORY, 0.8)]
    emb = AsyncMock()
    emb.embed.return_value = [[0.1]]

    out = await tool_get_categories(
        vector_store=vs, wiki_id="w", names=["Public", "Secret"], embedder=emb,
        allowed_namespaces=[NS_CATEGORY], user=user,
        access_client=_access_client({"Category:Secret"}),
    )

    assert out["found"] == ["Category:Public"]
    assert out["missing"] == ["Secret"]
    assert "Category:Secret" not in out["suggestions"]
    assert "Category:Other" in out["suggestions"]


async def test_category_members_truncation_from_unfiltered_fetch(user):
    client = SimpleNamespace(
        get_category_members=AsyncMock(return_value=[
            {"title": "A", "ns": 0}, {"title": "B", "ns": 0}, {"title": "C", "ns": 0},
        ]),
        check_read_access=AsyncMock(return_value={"A": True, "B": False, "C": False}),
    )
    out = await wiki_tools.tool_get_category_members("X", user, limit=3, client=client)
    assert [m["title"] for m in out["members"]] == ["A"]
    # The fetch hit the limit, so the filtered list is NOT the full set.
    assert out["truncated"] is True
    assert "full result set" not in out["note"]


# ---------------------------------------------------------------------
# argument bounds
# ---------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (None, 5), (0, 1), (-3, 1), (7, 7), (10_000, 50), ("12", 12), ("abc", 5), (True, 5),
])
async def test_vector_search_k_clamped(user, raw, expected):
    with patch.object(base, "tool_vector_search", AsyncMock(return_value={})) as tool:
        args = {"query": "q"} if raw is None else {"query": "q", "k": raw}
        await base.dispatch_tool_call("mw_vector_search", args, user, AsyncMock(), AsyncMock())
    assert tool.await_args.kwargs["k"] == expected


@pytest.mark.parametrize("tool_name,target,args", [
    ("mw_search_pages", "tool_search_pages", {"query": "q"}),
    ("mw_get_category_members", "tool_get_category_members", {"category": "C"}),
    ("mw_list_pages", "tool_list_pages", {}),
    ("mw_get_categories", "tool_get_categories", {}),
    ("mw_find_pages_by_title", "tool_find_pages_by_title", {"prefix": "P"}),
])
async def test_limits_clamped_to_500(user, tool_name, target, args):
    with patch.object(base, target, AsyncMock(return_value={})) as tool:
        await base.dispatch_tool_call(
            tool_name, {**args, "limit": 1_000_000}, user, AsyncMock(), AsyncMock(),
        )
    assert tool.await_args.kwargs["limit"] == 500


async def test_schema_tools_receive_user(user):
    with patch.object(base, "tool_list_pages", AsyncMock(return_value={})) as tool:
        await base.dispatch_tool_call("mw_list_pages", {}, user, AsyncMock(), AsyncMock())
    assert tool.await_args.kwargs["user"] is user


# ---------------------------------------------------------------------
# re-embedding on staleness
# ---------------------------------------------------------------------

async def test_staleness_uses_wiki_namespace_and_canonical_title(user):
    vs = AsyncMock()
    vs.get_embedding_last_modified.return_value = None
    client = SimpleNamespace(get_page_info=AsyncMock(return_value={
        "title": "Lab notes:Secret plan", "namespace": 3000, "pageid": 9,
    }))
    page = PageContent(wikitext="text", timestamp="20240115123000")

    with patch.object(wiki_tools, "embedding_queue") as queue:
        queue.enqueue = AsyncMock()
        await wiki_tools._check_embedding_staleness("lab notes:Secret plan", page, user, vs, client)

    job = queue.enqueue.await_args.args[0]
    assert job.namespace == 3000  # custom namespace, not guessed as 0
    assert job.title == "Lab notes:Secret plan"


async def test_staleness_prefers_page_response_fields(user):
    vs = AsyncMock()
    vs.get_embedding_last_modified.return_value = None
    client = SimpleNamespace(get_page_info=AsyncMock())
    page = PageContent(
        wikitext="text", timestamp="2024-01-15T12:30:00Z",
        title="Help:Intro", namespace=12, rev_id=77,
    )

    with patch.object(wiki_tools, "embedding_queue") as queue:
        queue.enqueue = AsyncMock()
        await wiki_tools._check_embedding_staleness("help:Intro", page, user, vs, client)

    job = queue.enqueue.await_args.args[0]
    assert (job.title, job.namespace, job.rev_id) == ("Help:Intro", 12, 77)
    client.get_page_info.assert_not_awaited()


async def test_staleness_skips_when_namespace_unknown(user):
    vs = AsyncMock()
    vs.get_embedding_last_modified.return_value = None
    client = SimpleNamespace(get_page_info=AsyncMock(return_value=None))
    page = PageContent(wikitext="text", timestamp="20240115123000")

    with patch.object(wiki_tools, "embedding_queue") as queue:
        queue.enqueue = AsyncMock()
        await wiki_tools._check_embedding_staleness("X", page, user, vs, client)

    queue.enqueue.assert_not_awaited()
