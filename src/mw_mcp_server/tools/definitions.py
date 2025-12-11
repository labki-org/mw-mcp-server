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
TOOL_MW_RUN_SMW_ASK: Final[str] = "mw_run_smw_ask"
TOOL_MW_VECTOR_SEARCH: Final[str] = "mw_vector_search"


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
            "name": TOOL_MW_RUN_SMW_ASK,
            "description": (
                "Executes a raw Semantic MediaWiki #ask query and returns the structured results. "
                "Use this for property-based queries and structured data retrieval."
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
                "Use this for open-ended natural language queries."
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
]
