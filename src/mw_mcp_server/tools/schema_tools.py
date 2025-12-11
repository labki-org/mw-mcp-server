"""
Schema Tools

This module implements tools for querying the vector index metadata to validation
schema elements (Categories, Properties without requiring a live MediaWiki query
or searching through full text.

They rely on the `FaissIndex` to filter pages by namespace.
"""

from typing import List, Optional, Any
from ..embeddings.index import FaissIndex

# MediaWiki Constants
NS_CATEGORY = 14
NS_PROPERTY = 102

async def tool_get_categories(
    faiss_index: FaissIndex,
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
        # Normalize input: Ensure no "Category:" prefix for lookup if stored with it? 
        # Actually our index stores full page titles. "Category:Foo".
        # Prompt definition says "without 'Category:' prefix". We must append it.
        
        valid_items = []
        # We need to scan efficiently or check existence. 
        # FaissIndex stores doc_map -> ID. It doesn't have a fast Title->Doc lookup 
        # unless we build it or iterate. Iterating is fine for 10-50 items.
        # But get_pages_by_namespace iterates anyway.
        
        # Optimization: Fetch all categories (filtered by prefix if possible, but here we have specific names)
        # Actually, get_pages_by_namespace is fast enough for thousands of pages (in-memory iteration).
        
        # Let's just fetch all categories once and check membership.
        all_cats = set(faiss_index.get_pages_by_namespace(NS_CATEGORY))
        
        for name in names:
            full_name = f"Category:{name}"
            if full_name in all_cats:
                valid_items.append(name) # Return as requested (without prefix? or with?)
                # Tool definition implies returning "list of existing". 
                # Let's return full titles to be unambiguous for the LLM.
                # Wait, description says "validates...". 
                # Prompts say "choose only identifiers that are confirmed".
                
        return sorted([f"Category:{n}" for n in valid_items])

    # Search Mode
    results = faiss_index.get_pages_by_namespace(NS_CATEGORY, pattern=prefix)
    return results[:limit]


async def tool_get_properties(
    faiss_index: FaissIndex,
    prefix: Optional[str] = None,
    names: Optional[List[str]] = None,
    limit: int = 50,
    **kwargs: Any,
) -> List[str]:
    """
    Retrieve existing property pages from the index.
    """
    if names:
        all_props = set(faiss_index.get_pages_by_namespace(NS_PROPERTY))
        valid_items = []
        for name in names:
            # Handle potential user confusion about "Has " vs "Property:Has "
            # We assume input is "Has name"
            full_name = f"Property:{name}"
            if full_name in all_props:
                valid_items.append(name)
        
        return sorted([f"Property:{n}" for n in valid_items])

    results = faiss_index.get_pages_by_namespace(NS_PROPERTY, pattern=prefix)
    return results[:limit]


async def tool_list_pages(
    faiss_index: FaissIndex,
    namespace: Optional[int] = None,
    prefix: Optional[str] = None,
    limit: int = 50,
    **kwargs: Any,
) -> List[str]:
    """
    Retrieve existing pages from the index for a given namespace.
    """
    results = faiss_index.get_pages_by_namespace(namespace, pattern=prefix)
    return results[:limit]
