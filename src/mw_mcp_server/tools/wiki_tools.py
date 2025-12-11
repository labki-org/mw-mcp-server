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

from ..wiki.api_client import MediaWikiClient
from ..wiki.smw_client import SMWClient
from ..auth.models import UserContext


# ---------------------------------------------------------------------
# Dependency Injection
# ---------------------------------------------------------------------

# These singletons are preserved for now, but we explicitly support
# dependency injection for testing or future multi-instance deployments.
mw_client = MediaWikiClient()
smw_client = SMWClient(mw_client)


# ---------------------------------------------------------------------
# Permission Layer (placeholder)
# ---------------------------------------------------------------------

def _assert_user_can_read(user: UserContext, title: str) -> None:
    """
    Placeholder for permission checks.

    Implementations could include:
    - Namespace-specific access rules
    - Role-based restrictions
    - ACL lookups
    - SMW-driven metadata checks
    """
    # At the moment, all users with a valid JWT may read.
    # This hook exists for future expansion.
    return


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

    _assert_user_can_read(user, title)

    client = client or mw_client

    try:
        text = await client.get_page_wikitext(title)
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

    Returns
    -------
    Dict[str, Any]
        Raw SMW ASK result structure.
    """
    if not ask_query:
        raise ValueError("mw_run_smw_ask requires a non-empty 'ask' argument.")

    client = client or smw_client

    try:
        result = await client.ask(ask_query)
    except Exception as exc:
        raise ValueError(
            f"SMW ASK query failed: {type(exc).__name__}: {str(exc)}"
        ) from exc

    # SMW responses are always dict-like
    if not isinstance(result, dict):
        raise ValueError("SMW ASK returned non-dict result.")

    return result
