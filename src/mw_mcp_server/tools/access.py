"""
Per-Page Read Filtering for Tool Output

Title listings built from the embedding index (or from fuzzy "did you
mean" matching against it) only know namespaces, not per-page restrictions
(Lockdown/ControlAccess). Every title such a tool returns to the LLM must
first be checked with MediaWiki's ``mwassistant-check-access`` endpoint for
the acting user.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

from ..auth.models import UserContext

if TYPE_CHECKING:
    from ..wiki.api_client import MediaWikiClient

# Titles checked per round while filling a listing.
ACCESS_CHECK_CHUNK = 100

# Upper bound on titles checked for a single listing, so a user who can read
# almost nothing cannot make one tool call fan out into unbounded requests.
MAX_ACCESS_CHECKED_TITLES = 1000


def _default_client() -> "MediaWikiClient":
    # Imported lazily: wiki_tools imports schema_tools, which imports us.
    from .wiki_tools import mw_client
    return mw_client


async def readable_subset(
    titles: Iterable[str],
    user: UserContext,
    client: Optional["MediaWikiClient"] = None,
) -> List[str]:
    """Return the titles (in order, deduplicated) the user can read."""
    unique = [t for t in dict.fromkeys(titles) if t]
    if not unique:
        return []
    client = client or _default_client()
    access: Dict[str, bool] = await client.check_read_access(unique, user)
    return [t for t in unique if access.get(t, False)]


async def filter_readable(
    candidates: Sequence[str],
    user: UserContext,
    limit: int,
    client: Optional["MediaWikiClient"] = None,
) -> Tuple[List[str], bool]:
    """
    Walk ``candidates`` in order and collect up to ``limit`` readable titles.

    Returns ``(readable, truncated)`` where ``truncated`` is True when more
    readable titles may exist beyond the returned ones (candidates were
    left unchecked, or more readable titles were found than ``limit``).
    """
    if limit < 1:
        return [], bool(candidates)

    unique = [t for t in dict.fromkeys(candidates) if t]
    client = client or _default_client()

    readable: List[str] = []
    pos = 0
    while pos < len(unique) and len(readable) <= limit and pos < MAX_ACCESS_CHECKED_TITLES:
        # Loop until limit + 1 readable titles are found, so we can tell
        # whether anything readable lies beyond the limit.
        size = min(ACCESS_CHECK_CHUNK, MAX_ACCESS_CHECKED_TITLES - pos)
        chunk = unique[pos:pos + size]
        pos += len(chunk)
        access = await client.check_read_access(chunk, user)
        readable.extend(t for t in chunk if access.get(t, False))

    truncated = len(readable) > limit or pos < len(unique)
    return readable[:limit], truncated
