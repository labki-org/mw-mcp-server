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
| `JWT_MW_TO_MCP_SECRET` | Shared with MWAssistant extension |
| `JWT_MCP_TO_MW_SECRET` | Shared with MWAssistant extension |
| `DB_PASSWORD` | PostgreSQL password |
| `DAILY_TOKEN_LIMIT` | Max tokens per user per day (default: 100,000) |

## Architecture

The server uses PostgreSQL with pgvector for:
- **Vector embeddings** - Semantic search over wiki content
- **Chat sessions** - Persistent conversation history per user
- **Token usage tracking** - Rate limiting based on daily token consumption

```
docker-compose.yml
├── mw-mcp-server (FastAPI)
└── postgres (pgvector/pgvector:pg16)
```

Data is persisted in Docker volumes:
- `pgdata` - PostgreSQL database files

## Rate Limiting

Users are limited to a configurable number of tokens per day (default: 100,000).
When a user exceeds their limit, they receive a 429 response with reset time.

Configure via `DAILY_TOKEN_LIMIT` environment variable:
```bash
# Allow 50,000 tokens per user per day (~$0.15/day at GPT-4o-mini prices)
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

One MCP server can serve multiple wikis. Each wiki needs:

1. **Unique `wiki_id`** in MWAssistant config
2. **Same JWT secrets** shared with MCP server

Data is isolated per-tenant within PostgreSQL tables.

### MediaWiki LocalSettings.php

```php
$wgMWAssistantMCPBaseUrl = 'https://mcp.example.com';
$wgMWAssistantWikiId = 'my-wiki';  // Unique per wiki
$wgMWAssistantJWTMWToMCPSecret = getenv('JWT_MW_TO_MCP_SECRET');
$wgMWAssistantJWTMCPToMWSecret = getenv('JWT_MCP_TO_MW_SECRET');
// Optional: Explicitly set public API URL if MCP cannot detect or reach standard URL
$wgMWAssistantWikiApiUrl = 'https://wiki.example.com/api.php';
```

## Monitoring

Check health:
```bash
curl http://localhost:8000/health
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
