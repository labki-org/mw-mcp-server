from functools import lru_cache
from typing import Generator

from ..llm.client import LLMClient
from ..embeddings.index import FaissIndex
from ..embeddings.embedder import Embedder

@lru_cache
def get_llm_client() -> LLMClient:
    return LLMClient()

@lru_cache
def get_faiss_index() -> FaissIndex:
    index = FaissIndex()
    # Attempt to load, similar to startup logic. 
    # Ideally, loading happens at startup, but for now we mirror current behavior.
    try:
        index.load()
    except Exception:
        pass
    return index

@lru_cache
def get_embedder() -> Embedder:
    return Embedder()
