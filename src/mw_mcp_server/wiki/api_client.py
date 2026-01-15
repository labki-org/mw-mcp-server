"""
MediaWiki API Client

This module provides a hardened, testable asynchronous client for interacting
with a MediaWiki API instance using short-lived MCP→MW JWT authentication.

Design Goals
------------
- One HTTP client per MediaWikiClient instance (connection pooling)
- Short-lived JWT per request with explicit scopes
- Deterministic failure behavior
- Strict response validation
- Dependency-injectable for tests
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
import logging
import httpx

from ..config import settings
from ..auth.jwt_utils import create_mcp_to_mw_jwt

logger = logging.getLogger("mcp.mediawiki")


# ---------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------

class MediaWikiClientError(RuntimeError):
    """Base exception for MediaWiki client failures."""


class MediaWikiRequestError(MediaWikiClientError):
    """Raised when an HTTP request to MediaWiki fails."""


class MediaWikiResponseError(MediaWikiClientError):
    """Raised when MediaWiki returns a malformed or unexpected response."""


# ---------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------

class MediaWikiClient:
    """
    Asynchronous MediaWiki API client authenticated via MCP→MW JWTs.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 15.0,
    ) -> None:
        """
        Parameters
        ----------
        base_url : Optional[str]
            MediaWiki API base URL. Defaults to settings.mw_api_base_url.

        timeout : float
            Per-request HTTP timeout in seconds.
        """
        self.base_url = base_url or str(settings.mw_api_base_url)
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # Lifecycle Management
    # ------------------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        """
        Lazily initialize and return an AsyncClient.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self._client

    async def aclose(self) -> None:
        """
        Close the underlying HTTP client.
        """
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Core Request Logic
    # ------------------------------------------------------------------

    async def _request(
        self,
        params: Dict[str, Any],
        scopes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform a single authenticated GET request to the MediaWiki API.

        Parameters
        ----------
        params : Dict[str, Any]
            MediaWiki API query parameters.

        scopes : Optional[List[str]]
            JWT scopes for this request. Defaults to ["page_read"].

        Returns
        -------
        Dict[str, Any]
            Parsed JSON response.

        Raises
        ------
        MediaWikiRequestError
            On transport-level failure or non-2xx status.

        MediaWikiResponseError
            On invalid or non-JSON responses.
        """
        if scopes is None:
            scopes = ["page_read"]

        token = create_mcp_to_mw_jwt(scopes)
        headers = {
            "Authorization": f"Bearer {token}",
        }

        client = await self._get_client()

        try:
            response = await client.get(
                self.base_url,
                params=params,
                headers=headers,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error(
                "MediaWiki request failed: %s %s (%s)",
                self.base_url,
                params,
                type(exc).__name__,
            )
            raise MediaWikiRequestError(
                f"MediaWiki request failed: {type(exc).__name__}"
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise MediaWikiResponseError(
                "MediaWiki returned non-JSON response."
            ) from exc

        return data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_page_wikitext(self, title: str) -> Optional[str]:
        """
        Fetch the raw wikitext of a MediaWiki page.

        Parameters
        ----------
        title : str
            Full page title including namespace if applicable.

        Returns
        -------
        Optional[str]
            Raw wikitext if the page exists, otherwise None.
        """
        if not title:
            raise ValueError("Page title must be non-empty.")

        params = {
            "action": "query",
            "prop": "revisions",
            "rvprop": "content",
            "titles": title,
            "format": "json",
            "formatversion": 2,
        }

        data = await self._request(params, scopes=["page_read"])

        try:
            pages = data["query"]["pages"]
        except KeyError as exc:
            raise MediaWikiResponseError(
                "Malformed MediaWiki response: missing query.pages."
            ) from exc

        if not pages or "missing" in pages[0]:
            return None

        try:
            return pages[0]["revisions"][0]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise MediaWikiResponseError(
                f"Malformed revision structure for page '{title}'."
            ) from exc

    async def get_all_pages(self, limit: int = 500) -> List[str]:
        """
        Return a list of all page titles in the main namespace.

        Parameters
        ----------
        limit : int
            Maximum number of pages to return in one call.

        Returns
        -------
        List[str]
            Page titles.
        """
        if limit < 1:
            raise ValueError("limit must be >= 1")

        params = {
            "action": "query",
            "list": "allpages",
            "aplimit": limit,
            "apnamespace": 0,
            "format": "json",
        }

        data = await self._request(params, scopes=["page_read"])

        try:
            pages = data["query"]["allpages"]
        except KeyError as exc:
            raise MediaWikiResponseError(
                "Malformed MediaWiki allpages response."
            ) from exc

        titles: List[str] = []

        for p in pages:
            title = p.get("title")
            if isinstance(title, str):
                titles.append(title)

        return titles

    async def search_pages(
        self, query: str, limit: int = 10, user: Optional[UserContext] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform a standard MediaWiki search via mwassistant-keyword-search.
        Allows impersonating a user given by 'user.username' and returns rich results.

        Parameters
        ----------
        query : str
            Search query string.
        
        limit : int
            Maximum number of results to return.

        user : Optional[UserContext]
            User context for impersonation.

        Returns
        -------
        List[Dict[str, Any]]
            List of search results with keys: title, snippet, size, wordcount, etc.
        """
        params = {
            "action": "mwassistant-keyword-search",
            "query": query,
            "limit": limit,
            "format": "json",
        }

        if user and user.username:
            params["username"] = user.username

        # scope "search" is required by the endpoint configuration
        data = await self._request(params, scopes=["search"])
        
        # The result is nested under 'mwassistant-keyword-search' key
        # We expect a list of dicts: {title, snippet, size, wordcount, timestamp}
        return data.get("mwassistant-keyword-search", [])

    async def check_read_access(
        self, titles: List[str], username: str
    ) -> Dict[str, bool]:
        """
        Check if a user can read each page title.

        This calls the mwassistant-check-access API endpoint to batch-validate
        read permissions against MediaWiki's permission system (including
        Lockdown and ControlAccess restrictions).

        Parameters
        ----------
        titles : List[str]
            List of page titles to check.

        username : str
            MediaWiki username to check permissions for.

        Returns
        -------
        Dict[str, bool]
            Map of page title to boolean indicating read access.
        """
        if not titles:
            return {}

        # Join titles with pipe as MW API convention
        params = {
            "action": "mwassistant-check-access",
            "titles": "|".join(titles),
            "username": username,
            "format": "json",
        }

        try:
            data = await self._request(params, scopes=["check_access"])
        except MediaWikiRequestError:
            # If the permission check fails, deny all access as a safe default
            logger.warning(
                "Permission check failed for user '%s' on %d titles",
                username,
                len(titles),
            )
            return {title: False for title in titles}

        # The result is nested under 'mwassistant-check-access' key
        result = data.get("mwassistant-check-access", {})
        access_map = result.get("access", {})

        # Ensure all requested titles have a response (default to False if missing)
        return {title: access_map.get(title, False) for title in titles}
