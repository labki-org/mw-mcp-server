# MediaWiki Extension Integration Guide

This document outlines how a MediaWiki extension should communicate with the `mw-mcp-server`.

## Authentication

This service uses **bidirectional short-lived JWT authentication** with a 30-second TTL. Two separate secrets are used:

- **`JWT_MW_TO_MCP_SECRET`**: Used by MWAssistant to sign tokens sent TO the MCP server
- **`JWT_MCP_TO_MW_SECRET`**: Used by MCP server to sign tokens sent TO MWAssistant

### MW → MCP (Extension calling Server)

All requests from MWAssistant to the MCP server must include a JWT in the `Authorization` header:

```
Authorization: Bearer <token>
```

#### Required JWT Claims (MW → MCP)

The token must be signed with `HS256` using the `JWT_MW_TO_MCP_SECRET`. It must include:

| Claim | Type | Required | Description |
|-------|------|----------|-------------|
| `iss` | string | Yes | Must be **"MWAssistant"** |
| `aud` | string | Yes | Must be **"mw-mcp-server"** |
| `iat` | number | Yes | Issued at timestamp (Unix epoch) |
| `exp` | number | Yes | Expiration timestamp (`iat + 30` seconds) |
| `user` | string | Yes | MediaWiki username of the user making the request |
| `roles` | array | Yes | List of MediaWiki user groups (e.g., `["user", "sysop"]`) |
| `scope` | array | Yes | List of operations this token grants access to |
| `client_id` | string | No | Identifier for the extension (default: "MWAssistant") |

**Example JWT Payload (MW → MCP):**
```json
{
  "iss": "MWAssistant",
  "aud": "mw-mcp-server",
  "iat": 1702345678,
  "exp": 1702345708,
  "user": "AdminUser",
  "roles": ["sysop", "bureaucrat"],
  "scope": ["chat_completion", "search"],
  "client_id": "MWAssistant"
}
```

#### Scope Requirements (MW → MCP)

Each endpoint requires specific scopes:

| Endpoint | Required Scope |
|----------|----------------|
| `POST /chat/` | `chat_completion` |
| `POST /search/` | `search` |
| `POST /smw-query/` | `smw_query` |
| `POST /actions/edit-page` | `edit_page` |

**Note:** The extension should include ALL scopes the user might need in a single token, as tokens are short-lived (30s).

### MCP → MW (Server calling Extension)

When the MCP server makes requests back to MediaWiki (e.g., to fetch page content), it includes a JWT signed with `JWT_MCP_TO_MW_SECRET`.

#### Required JWT Claims (MCP → MW)

| Claim | Type | Required | Description |
|-------|------|----------|-------------|
| `iss` | string | Yes | Must be **"mw-mcp-server"** |
| `aud` | string | Yes | Must be **"MWAssistant"** |
| `iat` | number | Yes | Issued at timestamp |
| `exp` | number | Yes | Expiration timestamp (`iat + 30`) |
| `scope` | array | Yes | Operations requested (e.g., `["page_read"]`) |

**Example JWT Payload (MCP → MW):**
```json
{
  "iss": "mw-mcp-server",
  "aud": "MWAssistant",
  "iat": 1702345678,
  "exp": 1702345708,
  "scope": ["page_read"]
}
```


## Endpoints

### 1. Chat Completion (`POST /chat/`)
Send a user message to get an AI response that may utilize wiki tools.

**Request:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hello, who are you?"
    }
  ],
  "session_id": "optional-uuid-v4-for-history",
  "max_tokens": 512
}
```

**Response:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hello, who are you?"
    },
    {
      "role": "assistant",
      "content": "I am the Wiki Assistant..."
    }
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
