"""
Tool Dispatch Layer

This module defines the central, authoritative dispatch mechanism for all
LLM-invoked tool calls. It enforces:

- Explicit tool allow-listing
- Strong argument validation
- Dependency injection for testability
- Uniform error behavior for the LLM tool loop

This is a critical security boundary. No tool should be callable unless it is
explicitly registered here.
"""

from __future__ import annotations

from typing import Any, Dict, Callable, Awaitable

from .wiki_tools import tool_get_page, tool_run_smw_ask
from .search_tools import tool_vector_search
from ..auth.models import UserContext
from ..embeddings.embedder import Embedder
from ..embeddings.index import FaissIndex


# ---------------------------------------------------------------------
# Tool Type Definitions
# ---------------------------------------------------------------------

ToolHandler = Callable[
    [Dict[str, Any], UserContext, FaissIndex, Embedder],
    Awaitable[Any],
]


# ---------------------------------------------------------------------
# Tool Registry (AUTHORITATIVE)
# ---------------------------------------------------------------------

async def _handle_get_page(
    args: Dict[str, Any],
    user: UserContext,
    faiss_index: FaissIndex,
    embedder: Embedder,
) -> Any:
    title = args.get("title")
    if not title:
        raise ValueError("mw_get_page requires 'title' argument.")
    return await tool_get_page(title, user)


async def _handle_smw_ask(
    args: Dict[str, Any],
    user: UserContext,
    faiss_index: FaissIndex,
    embedder: Embedder,
) -> Any:
    ask = args.get("ask")
    if not ask:
        raise ValueError("mw_run_smw_ask requires 'ask' argument.")
    return await tool_run_smw_ask(ask, user)


async def _handle_vector_search(
    args: Dict[str, Any],
    user: UserContext,
    faiss_index: FaissIndex,
    embedder: Embedder,
) -> Any:
    query = args.get("query")
    if not query:
        raise ValueError("mw_vector_search requires 'query' argument.")

    k = args.get("k", 5)

    return await tool_vector_search(
        query=query,
        user=user,
        faiss_index=faiss_index,
        embedder=embedder,
        k=k,
    )


TOOL_REGISTRY: Dict[str, ToolHandler] = {
    "mw_get_page": _handle_get_page,
    "mw_run_smw_ask": _handle_smw_ask,
    "mw_vector_search": _handle_vector_search,
}


# ---------------------------------------------------------------------
# Public Dispatch API
# ---------------------------------------------------------------------

async def dispatch_tool_call(
    tool_name: str,
    args: Dict[str, Any],
    user: UserContext,
    faiss_index: FaissIndex,
    embedder: Embedder,
) -> Any:
    """
    Dispatch a tool call requested by the LLM.

    Parameters
    ----------
    tool_name : str
        The symbolic tool name requested by the LLM.

    args : Dict[str, Any]
        Parsed JSON arguments for the tool.

    user : UserContext
        Authenticated user identity.

    faiss_index : FaissIndex
        Active FAISS index instance (injected).

    embedder : Embedder
        Active embedder instance (injected).

    Returns
    -------
    Any
        Tool execution result.

    Raises
    ------
    ValueError
        If the tool name is unknown or required arguments are missing.
    """

    handler = TOOL_REGISTRY.get(tool_name)
    if not handler:
        raise ValueError(f"Unknown tool requested: {tool_name}")

    return await handler(args, user, faiss_index, embedder)
