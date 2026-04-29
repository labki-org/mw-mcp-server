# MediaWiki Extension Integration Guide

This document outlines how a MediaWiki extension should communicate with the `mw-mcp-server`.

## Authentication

This service uses **bidirectional short-lived JWT authentication** with a 30-second TTL.
The server supports multiple tenants (wikis) via the `WIKI_CREDS` configuration. Each wiki has its own pair of secrets mapped by a `wiki_id`.

- **`mw_to_mcp_secret`**: Used by MWAssistant to sign tokens sent TO the MCP server.
- **`mcp_to_mw_secret`**: Used by MCP server to sign tokens sent TO MWAssistant.

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
| `wiki_id` | string | Yes | Unique identifier for the wiki (must match `WIKI_CREDS` on server) |
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
  "wiki_id": "my-corporate-wiki",
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

**Note:** The extension should include ALL scopes the user might need in a single token, as tokens are short-lived (30s).

### MCP → MW (Server calling Extension)

When the MCP server makes requests back to MediaWiki (e.g., to fetch page content), it includes a JWT signed with the `mcp_to_mw_secret` corresponding to the target wiki.

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

### 1b. Streaming Chat (`POST /chat/stream`)

Same request body as `POST /chat/`. Instead of buffering the full LLM tool loop
and returning a single JSON response, the server streams Server-Sent Events
(SSE) so the extension can render each step (tool start, tool result, assistant
turn) as it happens.

**Auth:** identical to `/chat/` — JWT in the `Authorization` header. The browser
`EventSource` API can't send custom headers, so the extension must use `fetch`
with a manual SSE parser:

```js
const res = await fetch(`${baseUrl}/chat/stream`, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${jwt}`,
    'Content-Type': 'application/json',
    'Accept': 'text/event-stream',
  },
  body: JSON.stringify({ messages, session_id }),
});

const reader = res.body.getReader();
const decoder = new TextDecoder();
let buf = '';
while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  buf += decoder.decode(value, { stream: true });
  const frames = buf.split('\n\n');
  buf = frames.pop();          // last fragment may be incomplete
  for (const frame of frames) {
    const lines = frame.split('\n');
    const event = lines.find(l => l.startsWith('event: '))?.slice(7);
    const data  = lines.find(l => l.startsWith('data: '))?.slice(6);
    if (event && data) handleEvent(event, JSON.parse(data));
  }
}
```

**Event sequence (always begins with `session`, always ends with `done` or `error`):**

| Event | Payload | Notes |
|-------|---------|-------|
| `session` | `{session_id, created}` | Always first. `created=true` if a new session was opened. Persist `session_id` immediately. |
| `assistant_message` | `{content, iteration, is_final}` | One per LLM iteration with text. The `is_final=true` one is the user-facing answer. |
| `tool_start` | `{call_id, name, args, iteration}` | Emitted right before each tool runs. Use `call_id` to correlate with `tool_result`. |
| `tool_result` | `{call_id, name, ok, result_preview, elapsed_ms}` | `ok=false` on tool error. `result_preview` is capped at ~4 KB; full result still fed back to the LLM. |
| `error` | `{code, message}` | Fatal error mid-stream. Stream closes after this. `code` is one of `llm_failure`, `internal`. |
| `done` | `{session_id, used_tools, tokens, final_content}` | Terminal event. `final_content` mirrors the last `assistant_message` for late-joining clients. |

**Example stream:**

```
event: session
data: {"session_id":"f3a1...","created":true}

event: tool_start
data: {"call_id":"call_abc","name":"mw_search_pages","args":{"query":"server room"},"iteration":0}

event: tool_result
data: {"call_id":"call_abc","name":"mw_search_pages","ok":true,"result_preview":[{"title":"Server_Room","score":0.92}],"elapsed_ms":143}

event: assistant_message
data: {"content":"The server room is in building 3.","iteration":1,"is_final":true}

event: done
data: {"session_id":"f3a1...","used_tools":[{"name":"mw_search_pages","args":"{\"query\":\"server room\"}","result":[...]}], "tokens":{"prompt":820,"completion":48,"total":868},"final_content":"The server room is in building 3."}
```

**Reverse-proxy notes:** SSE requires that nginx/Cloudflare not buffer the
response. The server already sends `X-Accel-Buffering: no` and
`Cache-Control: no-cache`, but nginx config in front of the server should set
`proxy_buffering off;` and `proxy_read_timeout 300s;` for the `/chat/stream`
location. Cloudflare's free tier buffers SSE — bypass via Page Rule or use
WebSockets if you can't disable that.

**Backwards compatibility:** `POST /chat/` is unchanged. Clients that don't
implement SSE keep working; only the streaming UI uses `/chat/stream`.

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

### 4. Embedding Update (`POST /embeddings/update`)
Push new or updated page embeddings to the server. Called by the MW extension when pages are created or edited.

**Request:**
```json
{
  "title": "Page_Title",
  "content": "Full wikitext content...",
  "namespace": 0,
  "last_modified": "20240115123000"
}
```

### 5. Embedding Delete (`POST /embeddings/delete`)
Remove embeddings when a page is deleted.

**Request:**
```json
{
  "title": "Page_Title"
}
```

### 6. Embedding Stats (`GET /embeddings/stats`)
Retrieve statistics about the embedding index for a wiki.

### 7. Schema Tools (via Chat Tool Calls)
The following tools are available to the LLM during chat and are invoked automatically:
- `mw_get_page` — Fetch raw wikitext for a page
- `mw_page_info` — Lightweight page metadata check
- `mw_vector_search` — Semantic vector search
- `mw_search_pages` — Standard MediaWiki keyword search
- `mw_run_smw_ask` — Execute SMW #ask queries
- `mw_get_categories` — List/validate categories from the index
- `mw_get_properties` — List/validate SMW properties from the index
- `mw_list_pages` — List pages in a namespace
- `mw_get_category_members` — List pages in a category

## Error Handling
- **401 Unauthorized**: Missing or invalid JWT.
- **403 Forbidden**: User lacks permission for the requested action or namespace.
- **500 Internal Error**: Server-side failure.
