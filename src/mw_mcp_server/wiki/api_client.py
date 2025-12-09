import httpx
from typing import Dict, Any
from ..config import settings

class MediaWikiClient:
    def __init__(self):
        self.base_url = str(settings.mw_api_base_url)

    async def _request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(self.base_url, params=params)
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
        data = await self._request(params)
        if "query" not in data:
            return None
        pages = data["query"].get("pages", [])
        if not pages or "missing" in pages[0]:
            return None
        return pages[0]["revisions"][0]["content"]

    async def create_or_edit_page(self, title: str, text: str, summary: str):
        # Placeholder for full edit workflow
        pass
