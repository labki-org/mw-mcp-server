"""
VectorStore Tests

LIKE-pattern escaping for title lookups, and the halfvec/HNSW search path
against a real PostgreSQL + pgvector (skipped when unavailable).
"""

from mw_mcp_server.config import settings
from mw_mcp_server.db.vector_store import VectorStore, escape_like

from pg_utils import pg_session_factory_fixture  # noqa: F401  (registers fixture)

WIKI = "test-wiki"


def test_escape_like_escapes_backslash_first():
    assert escape_like("50%_off\\x") == "50\\%\\_off\\\\x"
    assert escape_like("plain") == "plain"


async def _seed(sf, titles, namespace=0, wiki=WIKI):
    async with sf() as s:
        vs = VectorStore(s)
        dims = settings.embedding_dimensions
        vectors = [[float(i + 1)] + [0.0] * (dims - 1) for i in range(len(titles))]
        await vs.add_documents(
            wiki, titles, [None] * len(titles), [namespace] * len(titles), vectors
        )
        await s.commit()


async def test_pattern_is_matched_literally(pg_session_factory):
    await _seed(pg_session_factory, ["Has 100% Coverage", "Has 1000 Coverage", "A_B", "AxB"])

    async with pg_session_factory() as s:
        vs = VectorStore(s)
        assert await vs.get_pages_by_namespace(WIKI, pattern="100%") == ["Has 100% Coverage"]
        assert await vs.get_pages_by_namespace(WIKI, pattern="A_B") == ["A_B"]
        assert await vs.get_pages_by_namespace(WIKI, pattern="%") == ["Has 100% Coverage"]


async def test_search_filters_by_wiki_and_namespace(pg_session_factory):
    await _seed(pg_session_factory, [f"Other{i}" for i in range(20)], wiki="other-wiki")
    await _seed(pg_session_factory, ["Main1", "Main2"], namespace=0)
    await _seed(pg_session_factory, ["Cat1"], namespace=14)

    dims = settings.embedding_dimensions
    query = [1.0] + [0.0] * (dims - 1)
    async with pg_session_factory() as s:
        results = await VectorStore(s).search(WIKI, query, k=5, namespace_filter=[0])

    assert sorted(r[0] for r in results) == ["Main1", "Main2"]
    scores = [r[3] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert abs(scores[0] - 1.0) < 1e-6  # full-precision score, not halfvec
