from typing import Any, Dict
from .wiki_tools import tool_get_page, tool_run_smw_ask
from .search_tools import tool_vector_search
from ..embeddings.embedder import Embedder
from ..embeddings.index import FaissIndex

async def dispatch_tool_call(
    tool_name: str, 
    args: Dict[str, Any], 
    user_ctx,
    faiss_index: FaissIndex,
    embedder: Embedder
):
    if tool_name == "mw_get_page":
        return await tool_get_page(args["title"], user_ctx)
    elif tool_name == "mw_run_smw_ask":
        return await tool_run_smw_ask(args["ask"], user_ctx)
    elif tool_name == "mw_vector_search":
        return await tool_vector_search(
            query=args["query"],
            user=user_ctx,
            faiss_index=faiss_index,
            embedder=embedder,
            k=args.get("k", 5)
        )
    else:
        raise ValueError(f"Unknown tool: {tool_name}")
