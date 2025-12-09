from typing import List
import httpx
from ..config import settings

class Embedder:
    async def embed(self, texts: List[str]) -> List[list[float]]:
        payload = {"model": settings.embedding_model, "input": texts}
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.openai.com/v1/embeddings",
                json=payload,
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            )
        resp.raise_for_status()
        data = resp.json()
        return [d["embedding"] for d in data["data"]]
