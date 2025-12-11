from functools import lru_cache
from typing import Generator

from ..llm.client import LLMClient
from ..embeddings.index import FaissIndex
from ..embeddings.embedder import Embedder

@lru_cache
def get_llm_client() -> LLMClient:
    return LLMClient()

# Removed lru_cache to ensure we retry loading if it failed previously
# or if we want to ensure freshness. For performance, we should ideally use a singleton,
# but let's first fix the bug.
_global_index = None

def get_faiss_index() -> FaissIndex:
    global _global_index
    if _global_index and getattr(_global_index, "index", None) and _global_index.index.ntotal > 0:
        return _global_index
        
    index = FaissIndex()
    try:
        index.load()
        if getattr(index, "index", None) and index.index.ntotal > 0:
            _global_index = index
    except Exception:
        # If loading fails, we just return an empty index for now
        # and attempt reload on next valid request.
        pass
    
    return index

@lru_cache
def get_embedder() -> Embedder:
    return Embedder()
