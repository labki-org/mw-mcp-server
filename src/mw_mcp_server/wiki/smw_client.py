"""
Semantic MediaWiki (SMW) API Client

This module provides a hardened, testable client for executing Semantic
MediaWiki #ask queries via the MediaWiki API.

Design Goals
------------
- No dependence on private MediaWikiClient internals
- Deterministic request and response validation
- Explicit JWT scope usage
- Clean error semantics for LLM tool consumption
- Fully dependency-injectable for testing
"""

from __future__ import annotations

from typing import Dict, Any, Optional, TYPE_CHECKING
import hashlib
import logging

from .api_client import MediaWikiClient, MediaWikiRequestError, MediaWikiResponseError

if TYPE_CHECKING:
    from ..auth.models import UserContext

logger = logging.getLogger("mcp.smw")


# ---------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------

class SMWClientError(RuntimeError):
    """Base exception for Semantic MediaWiki client failures."""


class SMWQueryError(SMWClientError):
    """Raised when an SMW #ask query fails."""


# ---------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------

class SMWClient:
    """
    Semantic MediaWiki API client for executing #ask queries.
    """

    def __init__(self, mw_client: MediaWikiClient) -> None:
        """
        Parameters
        ----------
        mw_client : MediaWikiClient
            An initialized MediaWiki API client.
        """
        self._mw = mw_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ask(
        self,
        ask_query: str,
        params: Optional[Dict[str, Any]] = None,
        user: Optional["UserContext"] = None,
    ) -> Dict[str, Any]:
        """
        Execute a Semantic MediaWiki #ask query.

        Parameters
        ----------
        ask_query : str
            Raw SMW ask query string.

        params : Optional[Dict[str, Any]]
            Optional additional MediaWiki API parameters.

        user : Optional[UserContext]
            User context for permission-aware query execution.
            When provided, forwards username/user_id so the MW endpoint
            resolves the correct user for permission checks.

        Returns
        -------
        Dict[str, Any]
            The ``mwassistant-smw`` result object:
            ``{rows, printouts, count, has_more, errors, result}`` where
            ``rows`` is ``[{title, printouts: {label: [values]}}]`` (already
            filtered by the user's read permissions) and ``result`` is an
            escaped HTML table rendering.

        Raises
        ------
        ValueError
            If the query string is empty.

        SMWQueryError
            If the query fails at the transport or response level.
        """
        if not ask_query:
            raise ValueError("SMW ask query must be non-empty.")

        request_params: Dict[str, Any] = {
            "action": "mwassistant-smw",
            "query": ask_query,
            "format": "json",
            "formatversion": 2,
        }

        # Forward user identity for permission-aware query execution
        # (must match the signed user claims in the JWT).
        if user is not None and user.user_id:
            request_params["user_id"] = user.user_id
            if user.username:
                request_params["username"] = user.username

        if params:
            request_params.update(params)

        try:
            # We use 'smw_query' scope which matches the PHP requires_scopes check
            data = await self._mw.request(
                request_params,
                scopes=["smw_query"],
                api_url=user.api_url if user else None,
                wiki_id=user.wiki_id if user else None,
                user=user,
            )
        except (MediaWikiRequestError, MediaWikiResponseError) as exc:
            # Never log the query itself (it can contain user data).
            logger.error(
                "SMW ask query failed: len=%d sha256=%s (%s)",
                len(ask_query),
                hashlib.sha256(ask_query.encode("utf-8")).hexdigest()[:12],
                type(exc).__name__,
            )
            # API errors (e.g. unsupported printouts) are useful to the
            # caller; transport errors are not.
            detail = str(exc) if isinstance(exc, MediaWikiResponseError) else type(exc).__name__
            raise SMWQueryError(f"SMW query failed: {detail}") from exc

        result = data.get("mwassistant-smw")
        if not isinstance(result, dict):
            raise SMWQueryError("SMW query failed: malformed response from wiki.")
        return result
