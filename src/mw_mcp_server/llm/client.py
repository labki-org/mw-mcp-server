from typing import List, Dict, Any
import httpx
from ..config import settings

class LLMClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.openai_api_key

    async def chat(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Returns the raw message dict from OpenAI, e.g.:
        {
            "role": "assistant",
            "content": "...",
            "tool_calls": [...]
        }
        """
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]
