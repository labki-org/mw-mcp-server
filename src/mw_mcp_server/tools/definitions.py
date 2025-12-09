TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "mw_get_page",
            "description": "Fetches the raw wikitext content of a Semantic MediaWiki page by its title.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The exact title of the page to fetch, including namespace prefixes if any."
                    }
                },
                "required": ["title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "mw_run_smw_ask",
            "description": "Runs a Semantic MediaWiki #ask query to retrieve properties of pages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ask": {
                        "type": "string",
                        "description": "The SMW query string, e.g. '[[Category:City]]|?Population'."
                    }
                },
                "required": ["ask"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "mw_vector_search",
            "description": "Performs a semantic vector search over the wiki content to find relevant pages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, natural language or keywords."
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return (default 5)."
                    }
                },
                "required": ["query"]
            }
        }
    }
]
