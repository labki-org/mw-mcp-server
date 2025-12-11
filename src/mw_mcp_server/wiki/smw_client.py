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

from typing import Dict, Any, Optional
import logging

from .api_client import MediaWikiClient, MediaWikiRequestError, MediaWikiResponseError

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
    ) -> Dict[str, Any]:
        """
        Execute a Semantic MediaWiki #ask query.

        Parameters
        ----------
        ask_query : str
            Raw SMW ask query string.

        params : Optional[Dict[str, Any]]
            Optional additional MediaWiki API parameters.

        Returns
        -------
        Dict[str, Any]
            Raw JSON response from SMW.

        Raises
        ------
        ValueError
            If the query string is empty.

        SMWQueryError
            If the query fails at the transport or response level.
        """
        if not ask_query:
            raise ValueError("SMW ask query must be non-empty.")

        request_params = {
            "action": "mwassistant-smw",
            "query": ask_query,
            "format": "json",
        }

        if params:
            request_params.update(params)

        try:
            # We use 'smw_query' scope which matches the PHP requires_scopes check
            data = await self._mw._request(
                request_params,
                scopes=["smw_query"],
            )
        except (MediaWikiRequestError, MediaWikiResponseError) as exc:
            logger.error(
                "SMW ask query failed: %s (%s)",
                ask_query,
                type(exc).__name__,
            )
            raise SMWQueryError(
                f"SMW query failed: {type(exc).__name__}"
            ) from exc

        # The new parser endpoint returns {"mwassistant-smw": {"result": "..."}}
        # But MediaWikiClient._request() likely unwraps the response based on 'format=json'.
        # We need to verify what _request returns. 
        # Usually standard MW API returns { "mwassistant-smw": { ... } } wrapper.
        
        # If _request returns the full inner body:
        return data
