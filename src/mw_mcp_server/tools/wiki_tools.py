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

from typing import Optional, Dict, Any
import re

from ..wiki.api_client import MediaWikiClient
from ..wiki.smw_client import SMWClient
from ..auth.models import UserContext
from ..db import VectorStore
from .schema_tools import NS_CATEGORY, NS_PROPERTY


# ---------------------------------------------------------------------
# Dependency Injection
# ---------------------------------------------------------------------

# These singletons are preserved for now, but we explicitly support
# dependency injection for testing or future multi-instance deployments.
mw_client = MediaWikiClient()
smw_client = SMWClient(mw_client)


# ---------------------------------------------------------------------
# Permission Layer
# ---------------------------------------------------------------------

async def _assert_user_can_read(
    user: UserContext,
    title: str,
    client: Optional[MediaWikiClient] = None,
) -> None:
    """
    Validate that a user can read a specific page.

    This is called before returning page content to the LLM to ensure
    ControlAccess and Lockdown restrictions are respected.

    Parameters
    ----------
    user : UserContext
        Authenticated user context.

    title : str
        Page title to check.

    client : MediaWikiClient
        Optional client override for testing.

    Raises
    ------
    PermissionError
        If the user cannot read the page.
    """
    client = client or mw_client
    access_map = await client.check_read_access([title], user)

    if not access_map.get(title, False):
        raise PermissionError(
            f"User '{user.username}' does not have read access to page: {title}"
        )


# ---------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------

async def tool_get_page(
    title: str,
    user: UserContext,
    client: Optional[MediaWikiClient] = None,
) -> Optional[str]:
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

    Returns
    -------
    Optional[str]
        The wikitext of the page, or None if not found.
    """
    if not title:
        raise ValueError("mw_get_page requires a non-empty 'title' argument.")

    client = client or mw_client

    # Check permission before fetching page content
    await _assert_user_can_read(user, title, client)

    try:
        text = await client.get_page_wikitext(title, api_url=user.api_url, wiki_id=user.wiki_id, user=user)
    except Exception as exc:
        # Normalize all exceptions for LLM tool loop
        raise ValueError(
            f"Failed to fetch wiki page '{title}': {type(exc).__name__}"
        ) from exc

    return text


def _find_best_match(
    name: str,
    known_set: set,
    known_lower_map: dict,
    namespace_prefix: str,
) -> None:
    """
    Fuzzy-match *name* against a known set of wiki page titles.

    Raises ``ValueError`` with a suggestion when a close match is found.
    Returns silently when the name is valid or when no close match exists
    (to avoid false positives for built-in SMW properties, unindexed items, etc.).
    """
    tool_hint = (
        "mw_get_properties" if namespace_prefix == "Property:" else "mw_get_categories"
    )
    check_name = f"{namespace_prefix}{name}"

    # 1. Exact match — valid, nothing to do
    if check_name in known_set:
        return

    # 2. "Has " prefix variation (Property namespace only)
    if namespace_prefix == "Property:":
        has_variation = f"Property:Has {name}"
        if has_variation in known_set:
            raise ValueError(
                f"Property '{name}' does not exist. "
                f"Did you mean '{has_variation}'? "
                f"Please verify property names using `{tool_hint}`."
            )

    # 3. Case-insensitive match
    if check_name.lower() in known_lower_map:
        correct = known_lower_map[check_name.lower()]
        raise ValueError(
            f"{namespace_prefix.rstrip(':')}"
            f" '{name}' does not exist (Case Mismatch). "
            f"Did you mean '{correct}'? MediaWiki is case-sensitive."
        )

    # 4. Singular/plural heuristic
    if name.lower().endswith("s"):
        singular = name[:-1]
        sing_var = f"{namespace_prefix}{singular}"
        if sing_var in known_set:
            raise ValueError(
                f"{namespace_prefix.rstrip(':')} '{name}' not found. "
                f"Did you mean '{sing_var}'? "
                f"Please verify using `{tool_hint}`."
            )
    else:
        plural = name + "s"
        plural_var = f"{namespace_prefix}{plural}"
        if plural_var in known_set:
            raise ValueError(
                f"{namespace_prefix.rstrip(':')} '{name}' not found. "
                f"Did you mean '{plural_var}'? "
                f"Please verify using `{tool_hint}`."
            )


async def tool_run_smw_ask(
    ask_query: str,
    user: UserContext,
    client: Optional[SMWClient] = None,
    vector_store: Optional[VectorStore] = None,
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

    Returns
    -------
    Dict[str, Any]
        Raw SMW ASK result structure.
    """
    if not ask_query:
        raise ValueError("mw_run_smw_ask requires a non-empty 'ask' argument.")

    # Early deny: empty allowed_namespaces means user has no read access.
    # SMW does not respect Lockdown restrictions natively; the MW endpoint
    # post-filters results, but we can short-circuit here for users with
    # zero namespace access (consistent with vector search behaviour).
    if not user.allowed_namespaces:
        return {"result": "", "filtered_count": 0}

    if vector_store:
        # Fetch known properties and categories (one DB call each)
        known_props = set(await vector_store.get_pages_by_namespace(user.wiki_id, NS_PROPERTY))
        known_cats = set(await vector_store.get_pages_by_namespace(user.wiki_id, NS_CATEGORY))

        # Pre-build lowercase lookup maps once
        props_lower = {p.lower(): p for p in known_props}
        cats_lower = {c.lower(): c for c in known_cats}

        # A) Condition properties: [[PropertyName::Value]]
        for match in re.findall(r"\[\[([^:\]]+)::", ask_query):
            _find_best_match(match, known_props, props_lower, "Property:")

        # B) Category conditions: [[Category:Name]]
        #    Also catches OR syntax: [[Category:City||Category:Town]]
        for match in re.findall(r"Category:([^\]|]+)", ask_query):
            _find_best_match(match.strip(), known_cats, cats_lower, "Category:")

        # C) Printout properties: |?PropertyName  |?PropertyName=Label  |?PropertyName#fmt
        #    Skip special SMW printouts that are not real properties.
        SPECIAL_PRINTOUTS = {"category", "mainlabel"}
        for match in re.findall(r"\|\?([A-Za-z][^|=\]#]*)", ask_query):
            if match.strip().lower() not in SPECIAL_PRINTOUTS:
                _find_best_match(match.strip(), known_props, props_lower, "Property:")

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

    # Strip SMW result format parameters (e.g. |format=json, |format=csv).
    # Our API endpoint evaluates queries via the parser and extracts HTML
    # with getText(), so SMW format parameters like "json" produce empty
    # output.  The API already returns structured JSON; the SMW-level
    # format param is both unnecessary and breaks results.
    clean_query = re.sub(r"\|format\s*=\s*\w+", "", clean_query)

    try:
        result = await client.ask(clean_query, user=user)
    except Exception as exc:
        raise ValueError(
            f"SMW ASK query failed: {type(exc).__name__}: {str(exc)}"
        ) from exc

    if not isinstance(result, dict):
         return {"result": str(result)}

    return result
