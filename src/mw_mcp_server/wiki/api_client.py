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

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse
import asyncio
import logging
import httpx

from ..auth.jwt_utils import create_mcp_to_mw_jwt
from ..auth.models import UserContext

logger = logging.getLogger("mcp.mediawiki")

# mwassistant-check-access rejects more than this many titles per request.
CHECK_ACCESS_BATCH_SIZE = 100

# Request params that are safe to include in logs (everything else, e.g.
# usernames, page titles and queries, is redacted).
_LOGGABLE_PARAMS = frozenset({"action", "format", "formatversion", "prop", "list", "limit"})


def _redact_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of request params safe for logging."""
    return {
        key: (value if key in _LOGGABLE_PARAMS else "<redacted>")
        for key, value in params.items()
    }


def _user_params(user: Optional[UserContext]) -> Dict[str, Any]:
    """
    username/user_id params identifying the acting user to MWAssistant.

    These duplicate the signed ``user``/``user_id`` JWT claims (MWAssistant
    rejects a mismatch). For an anonymous identity (user_id 0) no username
    is sent, because MWAssistant compares it against the anonymous user's
    name and would reject it.
    """
    if user is None:
        return {}
    params: Dict[str, Any] = {}
    if user.user_id:
        params["user_id"] = user.user_id
        if user.username:
            params["username"] = user.username
    return params


# ---------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class PageContent:
    """
    Result of fetching page wikitext, optionally including revision timestamp.

    ``title``/``namespace``/``rev_id`` are filled when the wiki reports them
    (canonical prefixed title, namespace ID, latest revision ID).
    """
    wikitext: Optional[str]
    timestamp: Optional[str] = None
    title: Optional[str] = None
    namespace: Optional[int] = None
    rev_id: Optional[int] = None


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
        wiki_id: Optional[str] = None,
        timeout: float = 15.0,
    ) -> None:
        """
        Parameters
        ----------
        base_url : Optional[str]
            MediaWiki API base URL.
            
        wiki_id : Optional[str]
            Default Wiki ID for this client instance.

        timeout : float
            Per-request HTTP timeout in seconds.
        """
        self.base_url = base_url
        self.wiki_id = wiki_id
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
                # Never follow redirects: requests carry a signed JWT and the
                # target URL may come from a token claim.
                follow_redirects=False,
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


    async def request(
        self,
        params: Dict[str, Any],
        scopes: Optional[List[str]] = None,
        api_url: Optional[str] = None,
        wiki_id: Optional[str] = None,
        user: Optional[UserContext] = None,
        method: str = "GET",
    ) -> Dict[str, Any]:
        """
        Perform a single authenticated request to the MediaWiki API.

        Parameters
        ----------
        params : Dict[str, Any]
            MediaWiki API query parameters.

        scopes : Optional[List[str]]
            JWT scopes for this request. Defaults to ["page_read"].

        api_url : Optional[str]
            Override MediaWiki API URL for this request.
            
        wiki_id : Optional[str]
            Target wiki ID for signing the JWT request.

        user : Optional[UserContext]
            User the request acts for; bound into the JWT as signed
            ``user``/``user_id`` claims. None for system calls.

        method : str
            "GET" (default) or "POST" (params sent form-encoded).

        Returns
        -------
        Dict[str, Any]
            Parsed JSON response.
        """
        if scopes is None:
            scopes = ["page_read"]

        # Use passed wiki_id or fallback to instance default
        target_wiki_id = wiki_id or self.wiki_id
        
        target_url = api_url or self.base_url

        if not target_url:
            raise MediaWikiRequestError(
                "No API URL configured. The client must be initialized with a base_url "
                "or requests must provide a per-request 'api_url' via JWT."
            )

        parsed_url = urlparse(target_url)
        if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
            raise MediaWikiRequestError("MediaWiki API URL must be an absolute http(s) URL.")

        token = create_mcp_to_mw_jwt(scopes, wiki_id=target_wiki_id, user=user)
        headers = {
            "Authorization": f"Bearer {token}",
        }

        client = await self._get_client()

        try:
            if method.upper() == "POST":
                response = await client.post(target_url, data=params, headers=headers)
            else:
                response = await client.get(target_url, params=params, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error(
                "MediaWiki request failed: %s %s (%s)",
                parsed_url.netloc,
                _redact_params(params),
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

        # MediaWiki API often returns errors with HTTP 200.
        # Detect and surface them instead of silently returning partial data.
        if "error" in data:
            error_info = data["error"]
            code = error_info.get("code", "unknown")
            info = error_info.get("info", "No details provided")
            logger.error("MediaWiki API error: [%s] %s (action=%s)", code, info, params.get("action"))
            raise MediaWikiResponseError(
                f"MediaWiki API error [{code}]: {info}"
            )

        if "errors" in data:
            errors = data["errors"]
            if errors:
                first = errors[0] if isinstance(errors, list) else errors
                code = first.get("code", "unknown") if isinstance(first, dict) else "unknown"
                text = first.get("text", str(first)) if isinstance(first, dict) else str(first)
                logger.error("MediaWiki API error: [%s] %s (action=%s)", code, text, params.get("action"))
                raise MediaWikiResponseError(
                    f"MediaWiki API error [{code}]: {text}"
                )

        return data

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_first_page(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract the first page from a standard action=query response.

        Returns the page dict, or None if the page is missing.
        Raises MediaWikiResponseError if the response structure is malformed.
        """
        try:
            pages = data["query"]["pages"]
        except KeyError as exc:
            raise MediaWikiResponseError(
                "Malformed MediaWiki response: missing query.pages."
            ) from exc

        if not pages or "missing" in pages[0]:
            return None
        return pages[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_page_wikitext(
        self,
        title: str,
        api_url: Optional[str] = None,
        wiki_id: Optional[str] = None,
        user: Optional["UserContext"] = None,
    ) -> PageContent:
        """
        Fetch the raw wikitext of a MediaWiki page.

        When a user context is provided, uses the permission-aware
        mwassistant-page endpoint which respects Lockdown restrictions.
        Otherwise falls back to standard action=query (for admin/embedding scripts).

        Parameters
        ----------
        title : str
            Full page title including namespace if applicable.

        api_url : Optional[str]
            Target wiki API URL.

        wiki_id : Optional[str]
            Wiki identifier for JWT signing.

        user : Optional[UserContext]
            User context for permission-aware fetch. When provided,
            uses the mwassistant-page endpoint.

        Returns
        -------
        PageContent
            Wikitext and optional revision timestamp. Wikitext is None
            if the page does not exist or is not accessible.
        """
        if not title:
            raise ValueError("Page title must be non-empty.")

        # Permission-aware path: use mwassistant-page endpoint
        if user is not None:
            params: Dict[str, Any] = {
                "action": "mwassistant-page",
                "title": title,
                "format": "json",
                "formatversion": 2,
                **_user_params(user),
            }

            data = await self.request(
                params,
                scopes=["page_read"],
                api_url=user.api_url or api_url,
                wiki_id=user.wiki_id or wiki_id,
                user=user,
            )

            result = data.get("mwassistant-page", {})
            if result.get("error") == "permission-denied":
                raise PermissionError(
                    f"User '{user.username}' does not have read access to page: {title}"
                )
            if not result.get("exists", False):
                return PageContent(wikitext=None)
            ns = result.get("ns")
            rev_id = result.get("revid")
            canonical = result.get("title")
            return PageContent(
                wikitext=result.get("wikitext"),
                timestamp=result.get("timestamp"),
                title=canonical if isinstance(canonical, str) and canonical else None,
                namespace=ns if isinstance(ns, int) else None,
                rev_id=rev_id if isinstance(rev_id, int) and rev_id > 0 else None,
            )

        # Legacy path: standard action=query (no user permission checks)
        params = {
            "action": "query",
            "prop": "revisions",
            "rvprop": "content|timestamp",
            "titles": title,
            "format": "json",
            "formatversion": 2,
        }

        data = await self.request(params, scopes=["page_read"], api_url=api_url, wiki_id=wiki_id)
        page = self._extract_first_page(data)
        if page is None:
            return PageContent(wikitext=None)

        try:
            rev = page["revisions"][0]
            return PageContent(
                wikitext=rev["content"],
                timestamp=rev.get("timestamp"),
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise MediaWikiResponseError(
                f"Malformed revision structure for page '{title}'."
            ) from exc

    async def get_page_info(
        self,
        title: str,
        api_url: Optional[str] = None,
        wiki_id: Optional[str] = None,
        user: Optional["UserContext"] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch lightweight metadata about a page (existence, size, last modified, namespace).

        When a user context is provided, uses the permission-aware
        mwassistant-page-info endpoint which works on private wikis.

        Returns None if the page does not exist.
        """
        if not title:
            raise ValueError("Page title must be non-empty.")

        # Permission-aware path: use mwassistant-page-info endpoint
        if user is not None:
            params: Dict[str, Any] = {
                "action": "mwassistant-page-info",
                "title": title,
                "format": "json",
                "formatversion": 2,
                **_user_params(user),
            }

            data = await self.request(
                params,
                scopes=["page_read"],
                api_url=user.api_url or api_url,
                wiki_id=user.wiki_id or wiki_id,
                user=user,
            )

            result = data.get("mwassistant-page-info", {})
            if result.get("error") == "permission-denied":
                raise PermissionError(
                    f"User '{user.username}' does not have read access to page: {title}"
                )
            if not result.get("exists", False):
                return None

            return {
                "title": result.get("title", title),
                "pageid": result.get("pageid"),
                "namespace": result.get("ns"),
                "length": result.get("length", 0),
                "last_modified": result.get("timestamp"),
            }

        # Legacy path: standard action=query (no user permission checks)
        params = {
            "action": "query",
            "prop": "info",
            "titles": title,
            "format": "json",
            "formatversion": 2,
        }

        data = await self.request(
            params,
            scopes=["page_read"],
            api_url=api_url,
            wiki_id=wiki_id,
        )
        page = self._extract_first_page(data)
        if page is None:
            return None

        return {
            "title": page.get("title", title),
            "pageid": page.get("pageid"),
            "namespace": page.get("ns", 0),
            "length": page.get("length", 0),
            "last_modified": page.get("touched"),
        }

    async def get_page_revision_timestamp(
        self,
        title: str,
        api_url: Optional[str] = None,
        wiki_id: Optional[str] = None,
        user: Optional["UserContext"] = None,
    ) -> Optional[str]:
        """
        Fetch the latest revision timestamp for a page.

        Returns an ISO 8601 timestamp string, or None if the page does not exist.
        """
        if not title:
            raise ValueError("Page title must be non-empty.")

        params: Dict[str, Any] = {
            "action": "query",
            "prop": "revisions",
            "rvprop": "timestamp",
            "titles": title,
            "format": "json",
            "formatversion": 2,
        }

        data = await self.request(
            params,
            scopes=["page_read"],
            api_url=user.api_url if user else api_url,
            wiki_id=user.wiki_id if user else wiki_id,
            user=user,
        )
        page = self._extract_first_page(data)
        if page is None:
            return None

        try:
            return page["revisions"][0]["timestamp"]
        except (KeyError, IndexError, TypeError):
            return None

    async def get_category_members(
        self,
        category: str,
        limit: int = 50,
        api_url: Optional[str] = None,
        wiki_id: Optional[str] = None,
        user: Optional["UserContext"] = None,
    ) -> List[Dict[str, Any]]:
        """
        List pages in a given category.

        When a user context is provided, uses the permission-aware
        mwassistant-category-members endpoint which works on private wikis
        and filters results by the user's read permissions.

        Parameters
        ----------
        category : str
            Category name (with or without 'Category:' prefix).
        limit : int
            Maximum members to return.

        Returns
        -------
        List[Dict[str, Any]]
            List of dicts with keys: title, ns, pageid.
        """
        # Permission-aware path: use mwassistant-category-members endpoint
        if user is not None:
            params: Dict[str, Any] = {
                "action": "mwassistant-category-members",
                "category": category,
                "limit": min(max(limit, 1), 500),
                "format": "json",
                "formatversion": 2,
                **_user_params(user),
            }

            data = await self.request(
                params,
                scopes=["page_read"],
                api_url=user.api_url or api_url,
                wiki_id=user.wiki_id or wiki_id,
                user=user,
            )

            result = data.get("mwassistant-category-members", {})
            return result.get("members", [])

        # Legacy path: standard action=query
        if not category.startswith("Category:"):
            category = f"Category:{category}"

        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": min(limit, 500),
            "format": "json",
            "formatversion": 2,
        }

        data = await self.request(
            params,
            scopes=["page_read"],
            api_url=api_url,
            wiki_id=wiki_id,
        )

        try:
            return data["query"]["categorymembers"]
        except KeyError:
            return []

    async def get_all_pages(
        self,
        limit: int = 500,
        api_url: Optional[str] = None,
    ) -> List[str]:
        """
        Return a list of all page titles in the main namespace.

        Parameters
        ----------
        limit : int
            Maximum number of pages to return in one call.
        
        api_url : Optional[str]
            Target wiki API URL.

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

        data = await self.request(params, scopes=["page_read"], api_url=api_url)

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

    async def find_pages_by_title_prefix(
        self,
        prefix: str,
        namespace: int = 0,
        limit: int = 50,
        api_url: Optional[str] = None,
        wiki_id: Optional[str] = None,
        user: Optional["UserContext"] = None,
    ) -> List[Dict[str, Any]]:
        """
        List pages whose title starts with ``prefix`` via the permission-aware
        ``mwassistant-find-pages-by-title`` API module.

        We use the custom module instead of stock ``list=allpages`` because the
        latter is gated by MediaWiki's anonymous-read check on private wikis,
        which fires before any of our auth runs.

        Returns rows shaped ``{title, ns, pageid}``, already filtered through
        the user's read permissions.
        """
        if not prefix:
            raise ValueError("find_pages_by_title_prefix requires a non-empty prefix.")

        params: Dict[str, Any] = {
            "action": "mwassistant-find-pages-by-title",
            "prefix": prefix,
            "namespace": namespace,
            "limit": min(max(limit, 1), 500),
            "format": "json",
            "formatversion": 2,
            **_user_params(user),
        }

        target_api = user.api_url if user else api_url
        target_wiki = user.wiki_id if user else wiki_id
        data = await self.request(
            params, scopes=["page_read"], api_url=target_api, wiki_id=target_wiki,
            user=user,
        )

        return data.get("mwassistant-find-pages-by-title", {}).get("pages", [])

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
            **_user_params(user),
        }

        # scope "search" is required by the endpoint configuration
        data = await self.request(
            params,
            scopes=["search"],
            api_url=user.api_url if user else None,
            wiki_id=user.wiki_id if user else None,
            user=user,
        )
        
        # The result is nested under 'mwassistant-keyword-search' key
        # We expect a list of dicts: {title, snippet, size, wordcount, timestamp}
        return data.get("mwassistant-keyword-search", [])

    async def check_read_access(
        self, titles: List[str], user: UserContext | str
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

        user : UserContext | str
            MediaWiki user context (preferred) or username string to check permissions for.

        Returns
        -------
        Dict[str, bool]
            Map of page title to boolean indicating read access.
        """
        if not titles:
            return {}

        ctx = user if isinstance(user, UserContext) else None
        if ctx is not None:
            user_params = _user_params(ctx)
            api_url, wiki_id = ctx.api_url, ctx.wiki_id
        else:
            user_params = {"username": user} if user else {}
            api_url, wiki_id = None, None

        # The endpoint accepts at most CHECK_ACCESS_BATCH_SIZE titles per
        # request; POST keeps long title lists out of the URL.
        unique_titles = list(dict.fromkeys(titles))
        batches = [
            unique_titles[i:i + CHECK_ACCESS_BATCH_SIZE]
            for i in range(0, len(unique_titles), CHECK_ACCESS_BATCH_SIZE)
        ]

        async def check_batch(batch: List[str]) -> Dict[str, Any]:
            params = {
                "action": "mwassistant-check-access",
                "titles": "|".join(batch),
                "format": "json",
                "formatversion": 2,  # Proper JSON boolean serialization
                **user_params,
            }
            data = await self.request(
                params,
                scopes=["check_access"],
                api_url=api_url,
                wiki_id=wiki_id,
                user=ctx,
                method="POST",
            )
            # The result is nested under 'mwassistant-check-access' key
            return data.get("mwassistant-check-access", {}).get("access", {}) or {}

        access_map: Dict[str, Any] = {}
        for partial in await asyncio.gather(*(check_batch(b) for b in batches)):
            access_map.update(partial)

        # With formatversion=2, MW returns actual JSON booleans
        # Also handle legacy string format ('true'/'false') for backward compatibility
        def to_bool(val):
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() == "true"
            return False

        return {title: to_bool(access_map.get(title, False)) for title in titles}
