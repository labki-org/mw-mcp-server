from ..wiki.api_client import MediaWikiClient
from ..wiki.smw_client import SMWClient
from ..auth.models import UserContext

mw_client = MediaWikiClient()
smw_client = SMWClient(mw_client)

async def tool_get_page(title: str, user: UserContext) -> str | None:
    # Here one might check permissions
    text = await mw_client.get_page_wikitext(title)
    return text

async def tool_run_smw_ask(ask_query: str, user: UserContext) -> dict:
    return await smw_client.ask(ask_query)
