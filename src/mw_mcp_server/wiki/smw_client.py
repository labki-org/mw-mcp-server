from typing import Dict, Any, List
from .api_client import MediaWikiClient

class SMWClient:
    def __init__(self, mw_client: MediaWikiClient):
        self.mw = mw_client

    async def ask(self, ask_query: str, params: Dict[str, Any] | None = None) -> Dict:
        p = {
            "action": "ask",
            "query": ask_query,
            "format": "json",
        }
        if params:
            p.update(params)
        return await self.mw._request(p)
