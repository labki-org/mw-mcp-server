from fastapi import APIRouter, Depends
from typing import List
from .models import SearchRequest, SearchResult
from ..auth.security import require_scopes
from ..auth.models import UserContext
from ..tools.search_tools import tool_vector_search

router = APIRouter(prefix="/search", tags=["search"])

@router.post("/", response_model=List[SearchResult])
async def search(req: SearchRequest, user: UserContext = Depends(require_scopes("search"))):
    results = await tool_vector_search(req.query, user, req.k)
    return [SearchResult(**r) for r in results]

