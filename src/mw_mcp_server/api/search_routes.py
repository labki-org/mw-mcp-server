from fastapi import APIRouter, Depends
from typing import List
from .models import SearchRequest, SearchResult
from ..auth.security import get_current_user
from ..tools.search_tools import tool_vector_search

router = APIRouter(prefix="/search", tags=["search"])

@router.post("/", response_model=List[SearchResult])
async def search(req: SearchRequest, user=Depends(get_current_user)):
    results = await tool_vector_search(req.query, user, req.k)
    return [SearchResult(**r) for r in results]
