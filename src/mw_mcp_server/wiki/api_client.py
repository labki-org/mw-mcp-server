import httpx
from typing import Dict, Any
from ..config import settings
from ..auth.jwt_utils import create_mcp_to_mw_jwt

class MediaWikiClient:
    def __init__(self):
        self.base_url = str(settings.mw_api_base_url)

    async def _request(self, params: Dict[str, Any], scopes: list[str] = None) -> Dict[str, Any]:
        """
        Make authenticated request to MediaWiki API.
        
        Args:
            params: MediaWiki API parameters
            scopes: JWT scopes for this request (defaults to ["page_read"])
        """
        if scopes is None:
            scopes = ["page_read"]
        
        # Generate short-lived JWT for this request
        token = create_mcp_to_mw_jwt(scopes)
        headers = {
            "Authorization": f"Bearer {token}"
        }
        
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(self.base_url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()

    async def get_page_wikitext(self, title: str) -> str | None:
        params = {
            "action": "query",
            "prop": "revisions",
            "rvprop": "content",
            "titles": title,
            "format": "json",
            "formatversion": 2,
        }
        data = await self._request(params, scopes=["page_read"])
        if "query" not in data:
            return None
        pages = data["query"].get("pages", [])
        if not pages or "missing" in pages[0]:
            return None
        return pages[0]["revisions"][0]["content"]

    async def create_or_edit_page(self, title: str, text: str, summary: str):
        # Placeholder for full edit workflow
        # When implemented, should use scopes=["page_write"]
        pass

    async def get_all_pages(self, limit: int = 500) -> list[str]:
        """Returns a list of all page titles in the main namespace."""
        params = {
            "action": "query",
            "list": "allpages",
            "aplimit": limit,
            "apnamespace": 0,  # Main namespace
            "format": "json"
        }
        data = await self._request(params, scopes=["page_read"])
        pages = data.get("query", {}).get("allpages", [])
        return [p["title"] for p in pages]

