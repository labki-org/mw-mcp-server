"""
LLM Tool Definitions

This module defines the authoritative tool/function schemas exposed to the LLM.
These definitions must remain strictly synchronized with:

- tools/base.py (TOOL_REGISTRY)
- The actual tool handler implementations

This is a critical security boundary: only tools defined here can ever be
invoked by the LLM.
"""

from __future__ import annotations

from typing import Dict, List, Any, Final


# ---------------------------------------------------------------------
# Tool Name Constants (Single Source of Truth)
# ---------------------------------------------------------------------

TOOL_MW_GET_PAGE: Final[str] = "mw_get_page"
TOOL_MW_PAGE_INFO: Final[str] = "mw_page_info"
TOOL_MW_RUN_SMW_ASK: Final[str] = "mw_run_smw_ask"
TOOL_MW_VECTOR_SEARCH: Final[str] = "mw_vector_search"
TOOL_MW_SEARCH_PAGES: Final[str] = "mw_search_pages"
TOOL_MW_GET_CATEGORY_MEMBERS: Final[str] = "mw_get_category_members"
TOOL_MW_FIND_PAGES_BY_TITLE: Final[str] = "mw_find_pages_by_title"


# ---------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": TOOL_MW_GET_PAGE,
            "description": (
                "Fetches the raw wikitext content of a MediaWiki page by its title. "
                "This returns unrendered page source suitable for parsing or summarization."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": (
                            "Exact title of the page to fetch. "
                            "Include namespace prefixes if applicable (e.g., 'Category:Physics')."
                        ),
                        "minLength": 1,
                    }
                },
                "required": ["title"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_MW_PAGE_INFO,
            "description": (
                "Returns lightweight metadata about a wiki page: existence, namespace, size, and last modified date. "
                "Use this to check if a page exists without fetching its full content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": (
                            "Exact title of the page to check. "
                            "Include namespace prefixes if applicable (e.g., 'Category:Physics')."
                        ),
                        "minLength": 1,
                    }
                },
                "required": ["title"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_MW_GET_CATEGORY_MEMBERS,
            "description": (
                "Lists pages belonging to a given MediaWiki category. "
                "Returns a paginated `{members, count, limit, truncated, note, category}` envelope "
                "(see TRUNCATION AWARENESS in the system prompt)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": (
                            "Category name, with or without 'Category:' prefix "
                            "(e.g., 'Physics' or 'Category:Physics')."
                        ),
                        "minLength": 1,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of members to return.",
                        "minimum": 1,
                        "maximum": 500,
                        "default": 50,
                    },
                },
                "required": ["category"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_MW_RUN_SMW_ASK,
            "description": (
                "Executes a raw Semantic MediaWiki #ask query and returns the structured results. "
                "Use this for property-based queries and structured data retrieval. "
                "Note: |format= parameters are automatically stripped (the API returns structured JSON natively)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ask": {
                        "type": "string",
                        "description": (
                            "The SMW #ask query string, e.g. "
                            "'[[Category:City]]|?Population|?Country'."
                        ),
                        "minLength": 1,
                    }
                },
                "required": ["ask"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_MW_VECTOR_SEARCH,
            "description": (
                "Performs a semantic vector search over the embedded wiki content. "
                "Use this for open-ended natural language queries. "
                "Returns a paginated `{results, count, limit, truncated, note}` envelope "
                "(see TRUNCATION AWARENESS in the system prompt)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language or keyword search query.",
                        "minLength": 1,
                    },
                    "k": {
                        "type": "integer",
                        "description": (
                            "Maximum number of results to return. "
                            "Defaults to 5 if omitted."
                        ),
                        "minimum": 1,
                        "maximum": 50,
                        "default": 5,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_MW_SEARCH_PAGES,
            "description": (
                "MediaWiki FULLTEXT keyword search (list=search). Searches an "
                "asynchronously-built index, so brand-new pages can lag by minutes "
                "before appearing here. NOT authoritative for 'find pages whose title "
                "contains X' — use `mw_find_pages_by_title` for that. Best for content "
                "search ('pages that mention X'). Returns a paginated "
                "`{results, count, limit, truncated, note}` envelope."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword search query.",
                        "minLength": 1,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return.",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_MW_FIND_PAGES_BY_TITLE,
            "description": (
                "Find pages whose title starts with the given `prefix`, hitting the "
                "MediaWiki page table directly (`list=allpages`). Authoritative — "
                "newly-created pages appear immediately, no fulltext index lag. "
                "Use this for ANY 'find pages with X in the name / starting with X / "
                "newest page named X' question; do NOT use `mw_search_pages` for those. "
                "Returns a paginated `{results, count, limit, truncated, note, prefix, "
                "namespace}` envelope."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prefix": {
                        "type": "string",
                        "description": "Title prefix to match (case-sensitive).",
                        "minLength": 1,
                    },
                    "namespace": {
                        "type": "integer",
                        "description": (
                            "Namespace ID to search within. Default 0 (Main). "
                            "Common: 14=Category, 102=Property, 10=Template."
                        ),
                        "default": 0,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results.",
                        "minimum": 1,
                        "maximum": 500,
                        "default": 50,
                    },
                },
                "required": ["prefix"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mw_get_categories",
            "description": (
                "Look up category pages from the wiki index. "
                "Returns `{matches, suggestions, count, limit, truncated, note}` for prefix mode "
                "(see TRUNCATION AWARENESS in the system prompt), or "
                "`{found, missing, suggestions}` for names mode. The `suggestions` list contains "
                "semantically related categories found via vector search — when your literal term "
                "doesn't match (e.g. 'Lab member'), treat each suggestion as a candidate to "
                "investigate. An empty `matches`/`missing` entry does NOT prove the concept is absent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prefix": {
                        "type": "string",
                        "description": "Optional substring pattern to filter categories by.",
                    },
                    "names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of exact category names (without 'Category:' prefix) to check for existence.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return.",
                        "default": 50,
                        "maximum": 500,
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mw_get_properties",
            "description": (
                "Look up SMW property pages from the wiki index. "
                "Same response shape as `mw_get_categories`. The `suggestions` list contains "
                "semantically related properties — useful when properties use 'Has ' prefixes or "
                "naming conventions different from your literal query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prefix": {
                        "type": "string",
                        "description": "Optional substring pattern to filter properties by.",
                    },
                    "names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of exact property names (without 'Property:' prefix) to check for existence.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return.",
                        "default": 50,
                        "maximum": 500,
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mw_list_pages",
            "description": (
                "Lists existing pages in a specific namespace, optionally filtered by a name pattern. "
                "Use this to explore page titles or find pages when you know the namespace but not the exact name. "
                "Returns a paginated `{results, count, limit, truncated, note}` envelope "
                "(see TRUNCATION AWARENESS in the system prompt)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {
                        "anyOf": [
                            {"type": "integer"},
                            {"type": "string"}
                        ],
                        "description": (
                            "Client-side filter. Can be a namespace ID (e.g. 14) or name (e.g. 'Category', 'Property', 'Main'). "
                            "If omitted, searches all namespaces."
                        ),
                    },
                    "prefix": {
                        "type": "string",
                        "description": "Optional substring pattern to filter page titles by.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return.",
                        "default": 50,
                        "maximum": 500,
                    },
                },
                "additionalProperties": False,
            },
        },
    },
]
