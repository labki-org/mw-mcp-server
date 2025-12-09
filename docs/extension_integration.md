# MediaWiki Extension Integration Guide

This document outlines how a MediaWiki extension should communicate with the `mw-mcp-server`.

## Authentication

All requests to the MCP server must include a JWT in the `Authorization` header:

```
Authorization: Bearer <token>
```

### JWT Claims
The token must be signed with `HS256` using the `JWT_SECRET` shared between the extension and the server. It should include:

- `sub`: The MediaWiki username of the user initiating the request.
- `roles`: A list of user groups/roles (e.g., `["user", "sysop"]`).
- `client_id`: Identify the extension (e.g., `"mw-mcp-extension"`).

**Example Payload:**
```json
{
  "sub": "AdminUser",
  "roles": ["sysop", "bureaucrat"],
  "client_id": "mw_extension",
  "iat": 1715420000
}
```

## Endpoints

### 1. Chat Completion (`POST /chat/`)
Send a user message to get an AI response that may utilize wiki tools.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "What is the population of New York?"}
  ],
  "max_tokens": 512
}
```

**Response:**
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "The population is..."}
  ],
  "used_tools": []
}
```

### 2. Semantic Search (`POST /search/`)
Perform a vector search against the indexed wiki content.

**Request:**
```json
{
  "query": "project management tools",
  "k": 5
}
```

**Response:**
```json
[
  {
    "title": "Project_Management",
    "score": 0.85,
    "text": "..."
  }
]
```

### 3. SMW Query (`POST /smw-query/`)
Execute a raw Semantic MediaWiki `#ask` query.

**Request:**
```json
{
  "ask": "[[Category:City]]|?Population"
}
```

### 4. Edit Page (`POST /actions/edit-page`)
Apply an edit to a page on behalf of the bot user.

**Request:**
```json
{
  "title": "New_Page_Title",
  "new_text": "Page content...",
  "summary": "Created by AI"
}
```

## Error Handling
- **401 Unauthorized**: Missing or invalid JWT.
- **403 Forbidden**: User lacks permission for the requested action or namespace.
- **500 Internal Error**: Server-side failure.
