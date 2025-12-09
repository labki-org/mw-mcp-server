from ..wiki.api_client import MediaWikiClient
from ..auth.models import UserContext

mw_client = MediaWikiClient()

async def tool_apply_edit(title: str, new_text: str, summary: str, user: UserContext) -> dict:
    # Check if user has edit rights etc.
    # For now invoke mw_client
    await mw_client.create_or_edit_page(title, new_text, summary)
    return {"success": True, "new_rev_id": 0}

async def tool_propose_page(spec: dict, user: UserContext) -> str:
    # Generate draft logic
    return "Draft Content Based on Spec"
