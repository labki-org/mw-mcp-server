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
from .schema_tools import tool_get_categories, tool_get_properties, tool_list_pages
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
    return await tool_run_smw_ask(ask, user, faiss_index=faiss_index)


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


async def _handle_get_categories(
    args: Dict[str, Any],
    user: UserContext,
    faiss_index: FaissIndex,
    embedder: Embedder,
) -> Any:
    return await tool_get_categories(
        faiss_index=faiss_index,
        prefix=args.get("prefix"),
        names=args.get("names"),
        limit=args.get("limit", 50),
    )


async def _handle_get_properties(
    args: Dict[str, Any],
    user: UserContext,
    faiss_index: FaissIndex,
    embedder: Embedder,
) -> Any:
    return await tool_get_properties(
        faiss_index=faiss_index,
        prefix=args.get("prefix"),
        names=args.get("names"),
        limit=args.get("limit", 50),
    )


async def _handle_list_pages(
    args: Dict[str, Any],
    user: UserContext,
    faiss_index: FaissIndex,
    embedder: Embedder,
) -> Any:
    raw_ns = args.get("namespace")
    ns_id: Optional[int] = None

    if raw_ns is not None:
        # Try as integer first
        if isinstance(raw_ns, int):
            ns_id = raw_ns
        elif isinstance(raw_ns, str):
            if raw_ns.isdigit():
                ns_id = int(raw_ns)
            else:
                # Map common names to IDs (case-insensitive)
                mapping = {
                    "main": 0,
                    "talk": 1,
                    "user": 2,
                    "project": 4,
                    "file": 6,
                    "mediawiki": 8,
                    "template": 10,
                    "help": 12,
                    "category": 14,
                    "property": 102,
                }
                normalized = raw_ns.lower().strip()
                if normalized in mapping:
                    ns_id = mapping[normalized]
                else:
                    # Fallback: maybe the user provided a localized name or something unknown.
                    # For now, we error to be safe, or we could treat it as 0? Error is safer.
                    # Actually, let's try to verify if it's a known namespace in the index? 
                    # No, strict parsing is better.
                    raise ValueError(f"Unknown namespace alias: '{raw_ns}'. Use an ID or standard name (e.g. 'Category').")
        
    return await tool_list_pages(
        faiss_index=faiss_index,
        namespace=ns_id,
        prefix=args.get("prefix"),
        limit=args.get("limit", 50),
    )


TOOL_REGISTRY: Dict[str, ToolHandler] = {
    "mw_get_page": _handle_get_page,
    "mw_run_smw_ask": _handle_smw_ask,
    "mw_vector_search": _handle_vector_search,
    "mw_get_categories": _handle_get_categories,
    "mw_get_properties": _handle_get_properties,
    "mw_list_pages": _handle_list_pages,
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
