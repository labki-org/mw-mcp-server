"""
Schema Tools

This module implements tools for querying the vector index metadata to validate
schema elements (Categories, Properties) without requiring a live MediaWiki query
or searching through full text.

They rely on the `VectorStore` to filter pages by namespace.
"""

from typing import List, Optional, Any
from ..db import VectorStore

# MediaWiki Constants
NS_CATEGORY = 14
NS_PROPERTY = 102


async def tool_get_categories(
    vector_store: VectorStore,
    wiki_id: str,
    prefix: Optional[str] = None,
    names: Optional[List[str]] = None,
    limit: int = 50,
    **kwargs: Any,
) -> List[str]:
    """
    Retrieve existing category pages from the index.
    
    If `names` is provided, checks for existence of specific categories.
    Otherwise, lists categories matching `prefix`.
    """
    if names:
        # Validation Mode: Check specific names
        all_cats = set(await vector_store.get_pages_by_namespace(wiki_id, NS_CATEGORY))
        valid_items = []
        
        for name in names:
            full_name = f"Category:{name}"
            if full_name in all_cats:
                valid_items.append(name)
                
        return sorted([f"Category:{n}" for n in valid_items])

    # Search Mode
    results = await vector_store.get_pages_by_namespace(wiki_id, NS_CATEGORY, pattern=prefix)
    return results[:limit]


async def tool_get_properties(
    vector_store: VectorStore,
    wiki_id: str,
    prefix: Optional[str] = None,
    names: Optional[List[str]] = None,
    limit: int = 50,
    **kwargs: Any,
) -> List[str]:
    """
    Retrieve existing property pages from the index.
    """
    if names:
        all_props = set(await vector_store.get_pages_by_namespace(wiki_id, NS_PROPERTY))
        valid_items = []
        for name in names:
            full_name = f"Property:{name}"
            if full_name in all_props:
                valid_items.append(name)
        
        return sorted([f"Property:{n}" for n in valid_items])

    results = await vector_store.get_pages_by_namespace(wiki_id, NS_PROPERTY, pattern=prefix)
    return results[:limit]


async def tool_list_pages(
    vector_store: VectorStore,
    wiki_id: str,
    namespace: Optional[int] = None,
    prefix: Optional[str] = None,
    limit: int = 50,
    **kwargs: Any,
) -> List[str]:
    """
    Retrieve existing pages from the index for a given namespace.
    """
    results = await vector_store.get_pages_by_namespace(wiki_id, namespace, pattern=prefix)
    return results[:limit]
