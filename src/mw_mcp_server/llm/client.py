from typing import List, Dict, Any
import httpx
from ..config import settings

class LLMClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.openai_api_key

    async def chat(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
    ) -> str:
        # Defaulting to gpt-4o-mini as a modern efficient model
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "temperature": temperature,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
