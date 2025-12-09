from ..embeddings.embedder import Embedder
from ..embeddings.index import FaissIndex
from ..auth.models import UserContext
from ..config import settings

embedder = Embedder()
faiss_index = FaissIndex()

# Try loading existing index
try:
    faiss_index.load()
except Exception:
    pass

def _resolve_allowed_namespaces_for(user: UserContext) -> list[int]:
    # Parse setting "0,14" -> [0, 14]
    parts = settings.allowed_namespaces_public.split(",")
    return [int(x.strip()) for x in parts if x.strip().isdigit()]

def filter_by_permissions(
    results,
    user_ctx: UserContext,
) -> list:
    allowed_ns = _resolve_allowed_namespaces_for(user_ctx)
    return [(doc, score) for doc, score in results if doc.namespace in allowed_ns]

async def tool_vector_search(query: str, user: UserContext, k: int = 5):
    if not faiss_index.index or faiss_index.index.ntotal == 0:
        return []
    
    q_emb_list = await embedder.embed([query])
    if not q_emb_list:
        return []
    q_emb = q_emb_list[0]
    
    raw_results = faiss_index.search(q_emb, k)
    filtered = filter_by_permissions(raw_results, user)
    return [
        {
            "title": doc.page_title,
            "section_id": doc.section_id,
            "score": score,
            "text": doc.text[:400],
        }
        for doc, score in filtered
    ]
