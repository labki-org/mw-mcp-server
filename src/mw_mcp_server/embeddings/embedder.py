from typing import List
import httpx
import logging
from ..config import settings

logger = logging.getLogger("mcp")

class Embedder:
    async def embed(self, texts: List[str], batch_size: int = 20) -> List[list[float]]:
        # OpenAI has limits on the number of inputs and tokens.
        # Batching prevents 400 Bad Request for large pages.
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = {"model": settings.embedding_model, "input": batch}
            
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(
                        "https://api.openai.com/v1/embeddings",
                        json=payload,
                        headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                    )
                    
                if resp.status_code != 200:
                    logger.error(f"OpenAI Embedding Error ({resp.status_code}): {resp.text}")
                    resp.raise_for_status()
                
                data = resp.json()
                all_embeddings.extend([d["embedding"] for d in data["data"]])
                
            except httpx.HTTPStatusError as exc:
                # Re-raise with more context if possible, but logging above handles the detail.
                raise exc
            except Exception as exc:
                logger.error("Unexpected error during embedding", exc_info=exc)
                raise exc

        return all_embeddings
