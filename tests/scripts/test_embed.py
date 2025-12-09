import asyncio
import os
import sys
import httpx
# Ensure src is in pythonpath
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from dotenv import load_dotenv
load_dotenv()

from mw_mcp_server.config import settings

async def test_embed():
    print(f"Model: {settings.embedding_model}")
    print(f"API Key: {settings.openai_api_key[:5]}...{settings.openai_api_key[-3:]}")
    
    url = "https://api.openai.com/v1/embeddings"
    payload = {
        "model": settings.embedding_model,
        "input": ["Hello world"]
    }
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
    
    print(f"POST {url}")
    print(f"Payload: {payload}")
    
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload, headers=headers)
        print(f"Status: {resp.status_code}")
        print(f"Body: {resp.text}")

if __name__ == "__main__":
    asyncio.run(test_embed())
