#!/usr/bin/env python3
"""
Manual test script for OpenAI embedding API connectivity.

Usage:
    cd tests/scripts
    python manual_test_embed.py
"""
import asyncio
import os
import sys

import httpx

# Ensure src is in pythonpath
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Must load dotenv before importing settings
from dotenv import load_dotenv  # noqa: E402
load_dotenv()

from mw_mcp_server.config import settings  # noqa: E402


async def test_embed():
    api_key = settings.openai_api_key.get_secret_value()
    print(f"Model: {settings.embedding_model}")
    print(f"API Key: {api_key[:5]}...{api_key[-3:]}")
    
    url = "https://api.openai.com/v1/embeddings"
    payload = {
        "model": settings.embedding_model,
        "input": ["Hello world"]
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    
    print(f"POST {url}")
    print(f"Payload: {payload}")
    
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload, headers=headers)
        print(f"Status: {resp.status_code}")
        print(f"Body: {resp.text}")


if __name__ == "__main__":
    asyncio.run(test_embed())
