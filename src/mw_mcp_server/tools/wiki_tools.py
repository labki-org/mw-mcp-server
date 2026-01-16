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
        text = await client.get_page_wikitext(title, api_url=user.api_url)
    except Exception as exc:
        # Normalize all exceptions for LLM tool loop
        raise ValueError(
            f"Failed to fetch wiki page '{title}': {type(exc).__name__}"
        ) from exc

    return text


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

    if vector_store:
        # Extract potential property names: [[Property::Value]] or [[Property::...]]
        # Regex captures the part before '::' inside [[...]]
        matches = re.findall(r"\[\[([^:\]]+)::", ask_query)
        
        # NS_PROPERTY = 102
        known_props = set(await vector_store.get_pages_by_namespace(user.wiki_id, 102)) 
        
        # Build lookup maps if we have data
        if known_props:
            known_props_lower = {p.lower(): p for p in known_props}
            
            for prop_ref in matches:
                # 1. Normalize query reference to potential canonical Property page title
                if prop_ref.lower().startswith("property:"):
                    check_name = prop_ref
                else:
                    check_name = f"Property:{prop_ref}"

                # 2. Check exact existence
                if check_name in known_props:
                    continue
                
                # 3. Check for "Has " prefix variation (Classic SMW pattern)
                stripped_name = prop_ref.replace("Property:", "").strip()
                has_variation = f"Property:Has {stripped_name}"
                
                if has_variation in known_props:
                    raise ValueError(
                        f"Property '{prop_ref}' does not exist. "
                        f"Did you mean '{has_variation}'? "
                        "Please verify property names using `mw_get_properties`."
                    )

                # 4. Check for Case Insensitivity
                if check_name.lower() in known_props_lower:
                    correct = known_props_lower[check_name.lower()]
                    raise ValueError(
                        f"Property '{prop_ref}' does not exist (Case Mismatch). "
                        f"Did you mean '{correct}'? MediaWiki is case-sensitive."
                    )
                
                # 5. Check singular/plural (simple heuristic)
                if stripped_name.lower().endswith("s"):
                     singular = stripped_name[:-1]
                     sing_var = f"Property:{singular}"
                     if sing_var in known_props:
                         raise ValueError(f"Property '{prop_ref}' not found. Did you mean '{sing_var}'?")
                else:
                     plural = stripped_name + "s"
                     plural_var = f"Property:{plural}"
                     if plural_var in known_props:
                          raise ValueError(f"Property '{prop_ref}' not found. Did you mean '{plural_var}'?")

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

    try:
        result = await client.ask(clean_query)
    except Exception as exc:
        raise ValueError(
            f"SMW ASK query failed: {type(exc).__name__}: {str(exc)}"
        ) from exc

    if not isinstance(result, dict):
         return {"result": str(result)}

    return result
