from fastapi import APIRouter
from ..config import settings

router = APIRouter(tags=["health"])

@router.get("/health")
def health():
    return {"status": "ok", "mw_api": str(settings.mw_api_base_url)}
