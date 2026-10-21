"""
Wiki Tool Layer

This module defines LLM-callable tools that interact with the MediaWiki API
and Semantic MediaWiki (SMW).

Responsibilities
----------------
- Provide a safe tool API for LLM-invoked calls.
- Enforce permission checks (stubbed today, extendable later).
- Wrap MediaWikiClient and SMWClient operations with stable, predictable
  failure semantics for the LLM tool protocol.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Union
import asyncio
import re

import logging

from ..wiki.api_client import MediaWikiClient, PageContent
from ..wiki.smw_client import SMWClient
from ..auth.models import UserContext
from ..db import VectorStore
from ..embeddings.queue import embedding_queue, EmbeddingJob
from .schema_tools import NS_CATEGORY, NS_PROPERTY
from .pagination import paginated
from .access import readable_subset

logger = logging.getLogger("mcp.wiki_tools")

_SPECIAL_PRINTOUTS = frozenset({"category", "mainlabel"})

# The SMW tool result stays wrapped in this key: MWAssistant's chat UI
# looks for it to render the tool result as a table.
SMW_RESULT_KEY = "mwassistant-smw"
_REDIRECT_RE = re.compile(r"#REDIRECT\s*\[\[(.+?)\]\]", re.IGNORECASE)


# ---------------------------------------------------------------------
# Dependency Injection
# ---------------------------------------------------------------------

# These singletons are preserved for now, but we explicitly support
# dependency injection for testing or future multi-instance deployments.
mw_client = MediaWikiClient()
smw_client = SMWClient(mw_client)


# ---------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------

async def tool_get_page(
    title: str,
    user: UserContext,
    client: Optional[MediaWikiClient] = None,
    vector_store: Optional[VectorStore] = None,
) -> Union[str, Dict[str, Any]]:
    """
    Fetch raw wikitext for a MediaWiki page.

    Parameters
    ----------
    title : str
        Title of the wiki page to retrieve.

    user : UserContext
        Authenticated requesting user.

    client : MediaWikiClient
        Optional testing override.

    vector_store : VectorStore
        Optional vector store for staleness detection.

    Returns
    -------
    Union[str, Dict[str, Any]]
        The wikitext of the page, or a structured status dict if not found/empty.
    """
    if not title:
        raise ValueError("mw_get_page requires a non-empty 'title' argument.")

    client = client or mw_client

    try:
        page = await client.get_page_wikitext(title, api_url=user.api_url, wiki_id=user.wiki_id, user=user)
    except PermissionError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Failed to fetch wiki page '{title}': {type(exc).__name__}: {exc}"
        ) from exc

    if page.wikitext is None:
        return {"status": "not_found", "title": title}

    if not page.wikitext.strip():
        return {"status": "empty", "title": title, "content": ""}

    # Handle redirects: if the content is a redirect, follow it
    redirect_match = _REDIRECT_RE.match(page.wikitext)
    if redirect_match:
        target_title = redirect_match.group(1).strip()
        try:
            target_page = await client.get_page_wikitext(
                target_title, api_url=user.api_url, wiki_id=user.wiki_id, user=user,
            )
        except PermissionError:
            return {
                "status": "redirect",
                "title": title,
                "redirect_target": target_title,
                "content": page.wikitext,
                "note": "You do not have access to the redirect target page.",
            }
        except Exception:
            return {
                "status": "redirect",
                "title": title,
                "redirect_target": target_title,
                "content": page.wikitext,
                "note": "Failed to fetch redirect target.",
            }

        if target_page.wikitext is None:
            return {
                "status": "redirect",
                "title": title,
                "redirect_target": target_title,
                "content": page.wikitext,
                "note": "Redirect target page does not exist.",
            }

        # Staleness detection on the target page instead
        if vector_store is not None and target_page.wikitext:
            try:
                await _check_embedding_staleness(
                    target_title, target_page, user, vector_store, client
                )
            except Exception:
                logger.debug("Staleness check failed for '%s'", target_title, exc_info=True)

        return {
            "status": "redirect_followed",
            "original_title": title,
            "redirect_target": target_title,
            "content": target_page.wikitext,
        }

    # Staleness detection: compare live revision vs embedding timestamp
    if vector_store is not None and page.wikitext:
        try:
            await _check_embedding_staleness(
                title, page, user, vector_store, client
            )
        except Exception:
            # Staleness check is best-effort; don't fail the page fetch
            logger.debug("Staleness check failed for '%s'", title, exc_info=True)

    return page.wikitext


def _parse_revision_timestamp(rev_ts_str: str) -> datetime:
    """Parse a MW timestamp: ISO "2024-01-15T12:30:00Z" or MW format "20240115123000"."""
    if "T" in rev_ts_str:
        return datetime.fromisoformat(rev_ts_str.replace("Z", "+00:00"))
    return datetime.strptime(rev_ts_str, "%Y%m%d%H%M%S")


def _is_stale(rev_ts: datetime, emb_ts: Optional[datetime]) -> bool:
    if emb_ts is None:
        return True
    if emb_ts.tzinfo is None:
        return rev_ts.replace(tzinfo=None) > emb_ts
    return rev_ts > emb_ts


async def _check_embedding_staleness(
    title: str,
    page: PageContent,
    user: UserContext,
    vector_store: VectorStore,
    client: MediaWikiClient,
) -> None:
    """
    Compare page revision timestamp (from PageContent) against embedding timestamp.
    If stale or missing, enqueue a re-embedding job.

    The revision timestamp comes from the mwassistant-page response. The
    namespace and canonical title are taken from the wiki (mwassistant-page
    when it reports them, otherwise mwassistant-page-info) and never guessed
    from the title: a page in a custom namespace filed as namespace 0 would
    surface in main-namespace listings for users who cannot read it.
    """
    rev_ts_str = page.timestamp
    if rev_ts_str is None:
        return  # no timestamp available (older MW extension version)

    rev_ts = _parse_revision_timestamp(rev_ts_str)

    canonical = page.title or title
    emb_ts = await vector_store.get_embedding_last_modified(user.wiki_id, canonical)
    if not _is_stale(rev_ts, emb_ts):
        return

    namespace = page.namespace
    if namespace is None or page.title is None:
        info = await client.get_page_info(
            title, api_url=user.api_url, wiki_id=user.wiki_id, user=user,
        )
        info_title = info.get("title") if info else None
        info_ns = info.get("namespace") if info else None
        if not isinstance(info_title, str) or not info_title or not isinstance(info_ns, int):
            logger.debug("Skipping re-embed of '%s': wiki did not report namespace/title", title)
            return
        namespace = info_ns
        if info_title != canonical:
            canonical = info_title
            emb_ts = await vector_store.get_embedding_last_modified(user.wiki_id, canonical)
            if not _is_stale(rev_ts, emb_ts):
                return

    logger.info("Stale embedding detected for '%s', enqueuing refresh", canonical)
    await embedding_queue.enqueue(EmbeddingJob(
        wiki_id=user.wiki_id,
        title=canonical,
        content=page.wikitext,
        namespace=namespace,
        last_modified=rev_ts.replace(tzinfo=None),
        rev_id=page.rev_id,
        request_id=f"staleness-{user.wiki_id}-{canonical}",
    ))


def _parse_namespace_from_title(title: str) -> int:
    """Derive a namespace ID from a title prefix."""
    if ":" in title:
        prefix = title.split(":", 1)[0]
        ns_map = {"Category": 14, "Property": 102, "Template": 10, "Help": 12,
                   "User": 2, "File": 6, "MediaWiki": 8, "Talk": 1, "Project": 4}
        return ns_map.get(prefix, 0)
    return 0


async def tool_page_info(
    title: str,
    user: UserContext,
    client: Optional[MediaWikiClient] = None,
) -> Dict[str, Any]:
    """
    Return lightweight metadata about a page without fetching full content.

    Access control: checks allowed_namespaces from JWT, then check_read_access().
    If the user can't access the page, returns a permission error —
    does NOT reveal whether the page exists.
    """
    if not title:
        raise ValueError("mw_page_info requires a non-empty 'title' argument.")

    client = client or mw_client

    # Namespace-level check from JWT
    ns = _parse_namespace_from_title(title)
    if user.allowed_namespaces and ns not in user.allowed_namespaces:
        raise PermissionError(
            f"User '{user.username}' does not have access to namespace {ns}"
        )

    try:
        info = await client.get_page_info(
            title, api_url=user.api_url, wiki_id=user.wiki_id, user=user,
        )
    except PermissionError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Failed to fetch page info for '{title}': {type(exc).__name__}: {exc}"
        ) from exc

    if info is None:
        return {"status": "not_found", "title": title}

    return {
        "status": "exists",
        "title": info["title"],
        "namespace": info["namespace"],
        "length": info["length"],
        "last_modified": info["last_modified"],
    }


async def tool_get_category_members(
    category: str,
    user: UserContext,
    limit: int = 50,
    client: Optional[MediaWikiClient] = None,
) -> Dict[str, Any]:
    """
    List pages in a given category.

    Access control: requires NS_CATEGORY (14) in allowed_namespaces.
    Filters returned pages through check_read_access().
    """
    if not category:
        raise ValueError("mw_get_category_members requires a non-empty 'category' argument.")

    client = client or mw_client

    # Check that user can access the Category namespace
    if user.allowed_namespaces and NS_CATEGORY not in user.allowed_namespaces:
        raise PermissionError(
            f"User '{user.username}' does not have access to the Category namespace"
        )

    try:
        members = await client.get_category_members(
            category, limit=limit,
            api_url=user.api_url, wiki_id=user.wiki_id, user=user,
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to fetch category members for '{category}': {type(exc).__name__}: {exc}"
        ) from exc

    # Truncation is decided by the capped fetch, before filtering: a
    # filtered count below the limit does not mean the set is complete.
    hit_limit = len(members) >= limit

    # Filter through page-level access check
    if members:
        titles = [m.get("title", "") for m in members if m.get("title")]
        if titles:
            access_map = await client.check_read_access(titles, user)
            members = [m for m in members if access_map.get(m.get("title", ""), False)]

    rows = [{"title": m["title"], "ns": m.get("ns", 0)} for m in members]
    return paginated(
        rows,
        limit=limit,
        label="members",
        extra={"category": category},
        truncated=hit_limit,
    )


def _suggest_match(
    name: str,
    known_set: set,
    known_lower_map: dict,
    namespace_prefix: str,
) -> Optional[Tuple[str, str]]:
    """
    Fuzzy-match *name* against a known set of wiki page titles.

    Returns ``(suggested_title, error_message)`` when *name* does not exist
    but a close match does, else None. None is also returned when no close
    match exists (to avoid false positives for built-in SMW properties,
    unindexed items, etc.).
    """
    tool_hint = (
        "mw_get_properties" if namespace_prefix == "Property:" else "mw_get_categories"
    )
    check_name = f"{namespace_prefix}{name}"

    # 1. Exact match — valid, nothing to do
    if check_name in known_set:
        return None

    # 2. "Has " prefix variation (Property namespace only)
    if namespace_prefix == "Property:":
        has_variation = f"Property:Has {name}"
        if has_variation in known_set:
            return has_variation, (
                f"Property '{name}' does not exist. "
                f"Did you mean '{has_variation}'? "
                f"Please verify property names using `{tool_hint}`."
            )

    # 3. Case-insensitive match
    if check_name.lower() in known_lower_map:
        correct = known_lower_map[check_name.lower()]
        return correct, (
            f"{namespace_prefix.rstrip(':')}"
            f" '{name}' does not exist (Case Mismatch). "
            f"Did you mean '{correct}'? MediaWiki is case-sensitive."
        )

    # 4. Singular/plural heuristic
    variant = name[:-1] if name.lower().endswith("s") else name + "s"
    variant_full = f"{namespace_prefix}{variant}"
    if variant_full in known_set:
        return variant_full, (
            f"{namespace_prefix.rstrip(':')} '{name}' not found. "
            f"Did you mean '{variant_full}'? "
            f"Please verify using `{tool_hint}`."
        )

    return None


def _find_best_match(
    name: str,
    known_set: set,
    known_lower_map: dict,
    namespace_prefix: str,
) -> None:
    """
    Raise ``ValueError`` with a suggestion when *name* is a near miss of a
    known title (see :func:`_suggest_match`); return silently otherwise.

    Does not check whether the user may read the suggested page; callers
    exposing the message to a user must do that.
    """
    match = _suggest_match(name, known_set, known_lower_map, namespace_prefix)
    if match is not None:
        raise ValueError(match[1])


def _compact_smw_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reduce the mwassistant-smw result to what the LLM needs.

    Rows become ``{"title": ..., <label>: value-or-list}`` (a single value
    is unwrapped), and the HTML rendering is dropped.
    """
    printouts = [p for p in result.get("printouts") or [] if isinstance(p, str)]
    rows: List[Dict[str, Any]] = []
    for raw in result.get("rows") or []:
        if not isinstance(raw, dict):
            continue
        row: Dict[str, Any] = {"title": raw.get("title")}
        values_by_label = raw.get("printouts") or {}
        if isinstance(values_by_label, dict):
            for label, values in values_by_label.items():
                key = label if label != "title" else "title (printout)"
                if isinstance(values, list):
                    row[key] = values[0] if len(values) == 1 else values
                else:
                    row[key] = values
        rows.append(row)

    has_more = bool(result.get("has_more"))
    compact: Dict[str, Any] = {
        "rows": rows,
        "printouts": printouts,
        "count": result.get("count", len(rows)),
        "truncated": has_more,
        "errors": [e for e in result.get("errors") or [] if isinstance(e, str)],
    }
    if has_more:
        compact["note"] = (
            "More results exist than were returned. Narrow the query, or page "
            "through results with |limit= (max 500) and |offset=. Do NOT assume "
            "this is the full result set."
        )
    return compact


async def tool_run_smw_ask(
    ask_query: str,
    user: UserContext,
    client: Optional[SMWClient] = None,
    vector_store: Optional[VectorStore] = None,
    mw: Optional[MediaWikiClient] = None,
) -> Dict[str, Any]:
    """
    Execute a Semantic MediaWiki ASK query.

    Parameters
    ----------
    ask_query : str
        SMW ASK query string.

    user : UserContext
        Authenticated requesting user.

    client : SMWClient
        Optional testing override.
        
    vector_store : VectorStore
        Optional reference to vector store for schema validation.

    mw : MediaWikiClient
        Optional override for the client used to permission-check
        "did you mean" suggestions.

    Returns
    -------
    Dict[str, Any]
        ``{"mwassistant-smw": {rows, printouts, count, truncated, errors[, note]}}``
        where each row is ``{"title": ..., <printout label>: value-or-list}``.
    """
    if not ask_query:
        raise ValueError("mw_run_smw_ask requires a non-empty 'ask' argument.")

    # Early deny: empty allowed_namespaces means user has no read access.
    # SMW does not respect Lockdown restrictions natively; the MW endpoint
    # post-filters results, but we can short-circuit here for users with
    # zero namespace access (consistent with vector search behaviour).
    if not user.allowed_namespaces:
        return {SMW_RESULT_KEY: _compact_smw_result({})}

    if vector_store:
        # Extract references from the query before hitting the DB
        prop_conditions = re.findall(r"\[\[([^:\]]+)::", ask_query)
        cat_conditions = re.findall(r"Category:([^\]|]+)", ask_query)
        raw_printouts = re.findall(r"\|\?([A-Za-z][^|=\]#]*)", ask_query)
        printout_props = [
            m.strip() for m in raw_printouts
            if m.strip().lower() not in _SPECIAL_PRINTOUTS
        ]

        needs_props = bool(prop_conditions or printout_props)
        needs_cats = bool(cat_conditions)

        # Fetch only needed namespaces, concurrently when both are required
        known_props: set = set()
        known_cats: set = set()
        if needs_props and needs_cats:
            props_list, cats_list = await asyncio.gather(
                vector_store.get_pages_by_namespace(user.wiki_id, NS_PROPERTY),
                vector_store.get_pages_by_namespace(user.wiki_id, NS_CATEGORY),
            )
            known_props = set(props_list)
            known_cats = set(cats_list)
        elif needs_props:
            known_props = set(await vector_store.get_pages_by_namespace(user.wiki_id, NS_PROPERTY))
        elif needs_cats:
            known_cats = set(await vector_store.get_pages_by_namespace(user.wiki_id, NS_CATEGORY))

        # Pre-build lowercase lookup maps once
        props_lower = {p.lower(): p for p in known_props}
        cats_lower = {c.lower(): c for c in known_cats}

        suggestions: List[Tuple[str, str]] = []
        # A) Condition properties: [[PropertyName::Value]]
        # C) Printout properties: |?PropertyName  |?PropertyName=Label  |?PropertyName#fmt
        for match in prop_conditions + printout_props:
            suggestion = _suggest_match(match, known_props, props_lower, "Property:")
            if suggestion:
                suggestions.append(suggestion)

        # B) Category conditions: [[Category:Name]]
        for match in cat_conditions:
            suggestion = _suggest_match(match.strip(), known_cats, cats_lower, "Category:")
            if suggestion:
                suggestions.append(suggestion)

        # Only suggest pages the user can read: the index is filled
        # regardless of per-page restrictions.
        if suggestions:
            readable = set(await readable_subset(
                [title for title, _ in suggestions], user, client=mw or mw_client,
            ))
            for title, message in suggestions:
                if title in readable:
                    raise ValueError(message)

    client = client or smw_client
    
    # Simple strip if the LLM provided the full wrapper
    clean_query = ask_query.strip()
    if clean_query.startswith("{{#ask:") and clean_query.endswith("}}"):
        # Remove {{#ask: and trailing }}
        clean_query = clean_query[7:-2].strip()
    elif clean_query.startswith("{{") and clean_query.endswith("}}"):
         clean_query = clean_query[2:-2].strip()
         if clean_query.lower().startswith("#ask:"):
             clean_query = clean_query[5:].strip()

    # (|format= needs no stripping: the wiki ignores it and forces its own.)

    try:
        result = await client.ask(clean_query, user=user)
    except Exception as exc:
        raise ValueError(
            f"SMW ASK query failed: {type(exc).__name__}: {str(exc)}"
        ) from exc

    if not isinstance(result, dict):
        raise ValueError("SMW ASK query failed: unexpected response shape.")

    return {SMW_RESULT_KEY: _compact_smw_result(result)}
