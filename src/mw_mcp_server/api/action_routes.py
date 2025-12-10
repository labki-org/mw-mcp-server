from fastapi import APIRouter, Depends
from .models import EditRequest, EditResponse
from ..auth.security import require_scopes
from ..auth.models import UserContext
from ..tools.edit_tools import tool_apply_edit

router = APIRouter(prefix="/actions", tags=["actions"])

@router.post("/edit-page", response_model=EditResponse)
async def edit_page(req: EditRequest, user: UserContext = Depends(require_scopes("edit_page"))):
    result = await tool_apply_edit(req.title, req.new_text, req.summary, user)
    return EditResponse(**result)

