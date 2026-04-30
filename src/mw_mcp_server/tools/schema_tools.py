"""
Schema Tools

Tools for querying the vector index metadata to validate schema elements
(Categories, Properties) and to list pages in a given namespace.

Each tool returns a structured dict so the LLM receives both:
  - the explicit matches it asked about, and
  - semantic suggestions that may name the same concept under a different label.

The suggestions path is what catches cases like a user asking about "lab members"
on a wiki that uses "Researcher" / "Person" / "Organizational role" — without
them, the LLM tends to read "no exact match" as "concept absent" and stops.
"""

from typing import Any, Dict, List, Optional

from ..db import VectorStore
from ..embeddings.embedder import Embedder
from .pagination import paginated

# MediaWiki Constants
NS_CATEGORY = 14
NS_PROPERTY = 102

# How many semantic suggestions to surface when an exact lookup falls short.
SEMANTIC_SUGGESTION_LIMIT = 10

# Over-query pgvector by this multiple of k so dedupe-by-title still leaves
# us with k distinct suggestions when chunks of the same page rank highly.
SEMANTIC_OVERQUERY_MULTIPLIER = 3

# When prefix-mode returns fewer hits than this, attach semantic suggestions
# to help the LLM expand its search.
SUGGESTION_FALLBACK_THRESHOLD = 3


async def _semantic_namespace_suggestions(
    embedder: Embedder,
    vector_store: VectorStore,
    wiki_id: str,
    query: str,
    namespace: int,
    k: int = SEMANTIC_SUGGESTION_LIMIT,
    exclude: Optional[set] = None,
) -> List[str]:
    """Embed *query* and return up to k namespace-restricted vector hits.

    Used as a fallback when exact / prefix lookup misses, so the LLM still
    sees conceptually adjacent titles (e.g. 'Category:Person' for 'Lab member').
    """
    if not query:
        return []
    try:
        embeddings = await embedder.embed([query])
    except Exception:
        # Embedder failures shouldn't break the primary tool path.
        return []
    if not embeddings:
        return []

    raw = await vector_store.search(
        wiki_id=wiki_id,
        query_embedding=embeddings[0],
        k=k * SEMANTIC_OVERQUERY_MULTIPLIER,
        namespace_filter=[namespace],
    )

    exclude = exclude or set()
    seen: set = set()
    out: List[str] = []
    for title, _section, _ns, _score in raw:
        if title in seen or title in exclude:
            continue
        seen.add(title)
        out.append(title)
        if len(out) >= k:
            break
    return out


async def _list_namespace_with_suggestions(
    vector_store: VectorStore,
    embedder: Optional[Embedder],
    wiki_id: str,
    namespace: int,
    prefix_label: str,
    prefix: Optional[str],
    names: Optional[List[str]],
    limit: int,
    allowed_namespaces: Optional[List[int]],
) -> Dict[str, Any]:
    """Shared backbone for category / property lookups."""
    namespace_label = prefix_label.rstrip(":")

    # Permission gate: pretend the namespace is empty if user can't read it.
    if allowed_namespaces is not None and namespace not in allowed_namespaces:
        return paginated(
            [],
            limit=limit,
            label="matches",
            extra={
                "suggestions": [],
                "note": f"{namespace_label} namespace is not accessible to this user.",
            },
        )

    # ---- Names mode: existence check with semantic suggestions for misses ----
    if names:
        all_titles = set(await vector_store.get_pages_by_namespace(wiki_id, namespace))
        found: List[str] = []
        missing: List[str] = []
        for name in names:
            if f"{prefix_label}{name}" in all_titles:
                found.append(name)
            else:
                missing.append(name)

        suggestions: List[str] = []
        if missing and embedder is not None:
            suggestions = await _semantic_namespace_suggestions(
                embedder=embedder,
                vector_store=vector_store,
                wiki_id=wiki_id,
                query=" ".join(missing),
                namespace=namespace,
                exclude={f"{prefix_label}{n}" for n in found},
            )

        return {
            "found": sorted(f"{prefix_label}{n}" for n in found),
            "missing": missing,
            "suggestions": suggestions,
        }

    # ---- Prefix / list mode ----
    raw_matches = await vector_store.get_pages_by_namespace(
        wiki_id, namespace, pattern=prefix
    )
    matches = raw_matches[:limit]

    suggestions = []
    # Only run semantic fallback when the user gave us a query-shaped hint
    # (a prefix). A bare "list everything" call shouldn't pay for embedding.
    if prefix and len(matches) < SUGGESTION_FALLBACK_THRESHOLD and embedder is not None:
        suggestions = await _semantic_namespace_suggestions(
            embedder=embedder,
            vector_store=vector_store,
            wiki_id=wiki_id,
            query=prefix,
            namespace=namespace,
            exclude=set(matches),
        )

    extra: Dict[str, Any] = {"suggestions": suggestions}

    # Distinguish "namespace hasn't been indexed" from "prefix didn't match
    # anything" — the LLM can't tell from a bare empty list, but the user
    # cares about the difference (admin needs to run a batch embed).
    if not raw_matches and not prefix:
        extra["note"] = (
            f"The embedding index contains 0 pages in the {namespace_label} namespace. "
            "This usually means those pages haven't been embedded yet, NOT that the wiki "
            "has none — ask the wiki admin to run a batch embed for this namespace from "
            "Special:MWAssistantEmbeddings. To find pages by keyword regardless of the "
            "embedding index, use `mw_search_pages`."
        )

    return paginated(matches, limit=limit, label="matches", extra=extra)


async def tool_get_categories(
    vector_store: VectorStore,
    wiki_id: str,
    prefix: Optional[str] = None,
    names: Optional[List[str]] = None,
    limit: int = 50,
    allowed_namespaces: Optional[List[int]] = None,
    embedder: Optional[Embedder] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Look up category pages from the index.

    Returns ``{matches, suggestions}`` for prefix mode, or
    ``{found, missing, suggestions}`` for names mode.
    """
    return await _list_namespace_with_suggestions(
        vector_store=vector_store,
        embedder=embedder,
        wiki_id=wiki_id,
        namespace=NS_CATEGORY,
        prefix_label="Category:",
        prefix=prefix,
        names=names,
        limit=limit,
        allowed_namespaces=allowed_namespaces,
    )


async def tool_get_properties(
    vector_store: VectorStore,
    wiki_id: str,
    prefix: Optional[str] = None,
    names: Optional[List[str]] = None,
    limit: int = 50,
    allowed_namespaces: Optional[List[int]] = None,
    embedder: Optional[Embedder] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Look up property pages from the index. Same shape as tool_get_categories."""
    return await _list_namespace_with_suggestions(
        vector_store=vector_store,
        embedder=embedder,
        wiki_id=wiki_id,
        namespace=NS_PROPERTY,
        prefix_label="Property:",
        prefix=prefix,
        names=names,
        limit=limit,
        allowed_namespaces=allowed_namespaces,
    )


async def tool_list_pages(
    vector_store: VectorStore,
    wiki_id: str,
    namespace: Optional[int] = None,
    prefix: Optional[str] = None,
    limit: int = 50,
    allowed_namespaces: Optional[List[int]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Retrieve existing pages from the index for a given namespace.

    Returns a paginated wrapper ``{results, count, limit, truncated, note}``.
    """
    # Deny if user has no namespace access at all
    if allowed_namespaces is not None and not allowed_namespaces:
        return paginated(
            [],
            limit=limit,
            extra={"note": "User has no readable namespaces — nothing to list."},
        )

    # Deny if the requested namespace is not in user's allowed list
    if (
        allowed_namespaces is not None
        and namespace is not None
        and namespace not in allowed_namespaces
    ):
        return paginated(
            [],
            limit=limit,
            extra={
                "note": (
                    f"Namespace {namespace} is not accessible. "
                    f"Allowed namespace IDs: {sorted(allowed_namespaces)}."
                ),
            },
        )

    # Cross-namespace listing for a restricted user: aggregate over the
    # namespaces they CAN read instead of either denying outright (which
    # the LLM reads as 'wiki is empty') or letting the underlying call
    # leak pages from namespaces they can't read.
    if allowed_namespaces is not None and namespace is None:
        aggregated: List[str] = []
        per_ns_indexed = 0
        for ns in sorted(allowed_namespaces):
            if len(aggregated) >= limit:
                break
            rows = await vector_store.get_pages_by_namespace(
                wiki_id, ns, pattern=prefix
            )
            per_ns_indexed += len(rows)
            remaining = limit - len(aggregated)
            aggregated.extend(rows[:remaining])

        extra: Dict[str, Any] = {}
        if per_ns_indexed == 0 and not prefix:
            extra["note"] = (
                "The embedding index has 0 pages across this user's accessible "
                f"namespaces ({sorted(allowed_namespaces)}). This usually means "
                "pages haven't been embedded yet — ask the wiki admin to run a "
                "batch embed at Special:MWAssistantEmbeddings, or use "
                "`mw_search_pages` to find pages by keyword regardless of indexing."
            )
        return paginated(aggregated, limit=limit, extra=extra)

    results = await vector_store.get_pages_by_namespace(wiki_id, namespace, pattern=prefix)
    return paginated(results[:limit], limit=limit)
