# Deployment Guide

Deploy mw-mcp-server to a VPS with Caddy reverse proxy.

## Prerequisites

- Docker & Docker Compose
- Caddy (for SSL/reverse proxy)
- 2GB+ RAM recommended
- OpenAI API key

## Quick Start

```bash
# 1. Preparation
# Copy 'docker-compose.prod.yml' to your specific folder on the VPS
mkdir mw-mcp-server
cd mw-mcp-server
# (Upload docker-compose.prod.yml here and rename to docker-compose.yml if desired)

# 2. Configure Environment
# Copy .env.example to .env and fill in values
cp .env.example .env
nano .env

# 3. Deploy
# Pull the latest pre-built image from GitHub Container Registry
docker-compose -f docker-compose.prod.yml pull

# Start the service
docker-compose -f docker-compose.prod.yml up -d
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `WIKI_CREDS` | JSON map of wiki ID -> `{mw_to_mcp_secret, mcp_to_mw_secret, api_url?}` (see [Multi-Tenant Setup](#multi-tenant-setup)) |
| `DB_PASSWORD` | PostgreSQL password. Required: `docker-compose.prod.yml` refuses to start if unset |
| `DAILY_TOKEN_LIMIT` | Max tokens per user per day (default: 100,000) |
| `ADMIN_API_KEY` | Enables the admin usage dashboard (`/stats/dashboard`) |
| `ENABLE_API_DOCS` | `true` to expose `/docs`, `/redoc`, `/openapi.json`. Default off; leave off in production |
| `VERSION` | Image tag to run. CI pushes `latest` on **every** push to `main`, plus an immutable `sha-<short>` tag; pin `VERSION` to a `sha-` or release tag in production |

## Architecture

The server uses PostgreSQL with pgvector for:
- **Vector embeddings** - Semantic search over wiki content
- **Chat sessions** - Persistent conversation history per user
- **Token usage tracking** - Rate limiting based on daily token consumption

```
docker-compose.yml
├── mw-mcp-server (FastAPI)
└── postgres (pgvector/pgvector:0.8.7-pg16, pinned by digest)
```

The server runs as a **single uvicorn worker** on purpose: the embedding job
queue is an in-process `asyncio.Queue` (not persisted), and startup runs DB
migrations. Consequences:

- Embedding jobs that are queued but not yet processed are **lost on restart**;
  MWAssistant re-sends page updates on the next edit (or run a full reindex).
- Do not scale out with multiple workers/replicas until the queue is moved to
  shared storage.

When upgrading an existing database whose `vector` extension predates 0.7,
run `ALTER EXTENSION vector UPDATE;` (the image ships 0.8.7, but an existing
volume keeps the version it was created with).

Data is persisted in Docker volumes:
- `pgdata` - PostgreSQL database files

## Rate Limiting

Users are limited to a configurable number of tokens per day (default: 100,000).
When a user exceeds their limit, they receive a 429 response with reset time.

Configure via `DAILY_TOKEN_LIMIT` environment variable:
```bash
# Allow 50,000 tokens per user per day (~$0.40/day at GPT-4.1-mini prices)
DAILY_TOKEN_LIMIT=50000
```

View usage in the database:
```sql
SELECT wiki_id, user_id, usage_date, total_tokens, request_count 
FROM token_usage 
ORDER BY usage_date DESC;
```

## Caddy Configuration

Add to your Caddyfile:

```caddyfile
# Option 1: Dedicated subdomain
mcp.example.com {
    reverse_proxy localhost:8000
}

# Option 2: Path-based routing on wiki domain
wiki.example.com {
    handle /mcp-api/* {
        uri strip_prefix /mcp-api
        reverse_proxy localhost:8000
    }
    
    # Rest of wiki config...
    reverse_proxy localhost:8080
}
```

Reload Caddy: `sudo systemctl reload caddy`

## Multi-Tenant Setup

One MCP server can serve multiple wikis. Each wiki is isolated by a unique `wiki_id`.

**Server Configuration (`.env`):**
```bash
WIKI_CREDS='{
  "wiki-id-1": {
    "mw_to_mcp_secret": "long-secret-for-wiki-1-inbound",
    "mcp_to_mw_secret": "long-secret-for-wiki-1-outbound",
    "api_url": "https://wiki1.example.com/api.php"
  },
  "wiki-id-2": {
    "mw_to_mcp_secret": "long-secret-for-wiki-2-inbound",
    "mcp_to_mw_secret": "long-secret-for-wiki-2-outbound"
  }
}'
```

(Write it on a single line in a real `.env`; multi-line values break Docker Compose.)

`api_url` (optional, **recommended**) pins the wiki's `api.php` endpoint on the
server. When set, the MCP server only ever calls that URL for this tenant and
ignores any API URL supplied by the wiki, so a leaked `mw_to_mcp_secret` cannot
be used to point the server (and its signed MCP->MW tokens) at another host.
When omitted, the previous behaviour (URL provided by the wiki) is kept for
backward compatibility.

### MCP -> MW token claims

Tokens the server sends back to MWAssistant now also carry `user`, `user_id`
and `wiki_id` claims identifying the end user and tenant the call is made for.
MWAssistant shipped with labki-platform >= v0.4.2 enforces them (the wiki checks
that the token was issued for itself and acts as that user). Older MWAssistant
versions only validate `iss`/`aud`/`iat`/`exp`/`scope` and ignore unknown
claims, so mixed versions keep working, without the extra protection.

### MediaWiki LocalSettings.php (for "wiki-id-1")

```php
$wgMWAssistantMCPBaseUrl = 'https://mcp.example.com';
$wgMWAssistantWikiId = 'wiki-id-1';
$wgMWAssistantJWTMWToMCPSecret = 'long-secret-for-wiki-1-inbound';
$wgMWAssistantJWTMCPToMWSecret = 'long-secret-for-wiki-1-outbound';
// Optional: Explicitly set public API URL if MCP cannot detect or reach standard URL
$wgMWAssistantWikiApiUrl = 'https://wiki.example.com/api.php';
```

## Monitoring

Check health (`/health` = liveness, `/health/ready` = database reachable):
```bash
docker compose -f docker-compose.prod.yml ps   # shows (healthy)/(unhealthy)
docker exec mw-mcp-server python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health/ready').read())"
```
The image has no `curl`; the container healthcheck uses Python's `urllib`
against `/health/ready`.

## Admin Dashboard

Set `ADMIN_API_KEY`, then open `https://mcp.example.com/stats/dashboard` and
enter the key. The key is sent only in the `x-admin-key` header (the old
`?key=` query parameter is no longer accepted, since URLs end up in access
logs and browser history) and is kept in `sessionStorage` for the current tab
only. Scripted access:
```bash
curl -H "x-admin-key: $ADMIN_API_KEY" https://mcp.example.com/stats/usage?days=30
```

View logs:
```bash
docker logs mw-mcp-server -f
docker logs mcp-postgres -f
```

## Database Access

Connect to PostgreSQL:
```bash
docker exec -it mcp-postgres psql -U mcp -d mcp
```

View tables:
```sql
\dt                          -- List tables
SELECT COUNT(*) FROM embedding;  -- Embedding count
SELECT COUNT(*) FROM chat_session;  -- Session count
```

## Backup

Backup PostgreSQL:
```bash
docker exec mcp-postgres pg_dump -U mcp mcp > backup.sql
```

Restore:
```bash
cat backup.sql | docker exec -i mcp-postgres psql -U mcp mcp
```
