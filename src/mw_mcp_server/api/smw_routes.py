from fastapi import APIRouter, Depends
from .models import SMWQueryRequest, SMWQueryResponse
from ..auth.security import get_current_user
from ..tools.wiki_tools import tool_run_smw_ask

router = APIRouter(prefix="/smw-query", tags=["smw"])

@router.post("/", response_model=SMWQueryResponse)
async def smw_query(req: SMWQueryRequest, user=Depends(get_current_user)):
    data = await tool_run_smw_ask(req.ask, user)
    return SMWQueryResponse(raw=data)
