from fastapi import APIRouter, Depends, HTTPException, Body
from .models import EmbeddingUpdatePageRequest, EmbeddingDeletePageRequest, EmbeddingStatsResponse
from ..auth.security import require_scopes
from ..auth.models import UserContext
from ..embeddings.models import IndexedDocument
from ..tools.search_tools import faiss_index, embedder

router = APIRouter(prefix="/embeddings", tags=["embeddings"])

@router.get("/stats", response_model=EmbeddingStatsResponse)
async def get_embedding_stats(user: UserContext = Depends(require_scopes("embeddings"))):
    return faiss_index.get_stats()

@router.post("/page")
async def update_page_embedding(
    req: EmbeddingUpdatePageRequest, 
    user: UserContext = Depends(require_scopes("embeddings"))
):
    # 1. Chunk content
    # Simple chunking logic from original script
    text = req.content
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    docs = []
    texts_to_embed = []
    
    for para in paragraphs:
        if len(para) < 50:
            continue
        texts_to_embed.append(para)
        docs.append(IndexedDocument(
            page_title=req.title,
            text=para,
            namespace=0, # Defaulting to main namespace for now
            last_modified=req.last_modified
        ))
        
    # 2. Delete existing
    faiss_index.delete_page(req.title)
    
    if not docs:
        faiss_index.save()
        return {"status": "deleted_empty", "count": 0}

    # 3. Embed & Add
    try:
        embeddings = await embedder.embed(texts_to_embed)
        faiss_index.add_documents(docs, embeddings)
        faiss_index.save()
        return {"status": "updated", "count": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/page")
async def delete_page_embedding(
    req: EmbeddingDeletePageRequest,
    user: UserContext = Depends(require_scopes("embeddings"))
):
    count = faiss_index.delete_page(req.title)
    faiss_index.save()
    return {"status": "deleted", "count": count}
