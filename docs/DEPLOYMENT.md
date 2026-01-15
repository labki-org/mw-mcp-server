# Deployment Guide

Deploy mw-mcp-server to a VPS with Caddy reverse proxy.

## Prerequisites

- Docker & Docker Compose
- Caddy (for SSL/reverse proxy)
- 2GB+ RAM recommended
- OpenAI API key

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/labki-org/mw-mcp-server.git
cd mw-mcp-server
cp .env.example .env

# 2. Generate secrets
./scripts/generate-secrets.sh >> .env
# Or manually:
# openssl rand -base64 48

# 3. Edit .env with your values
nano .env

# 4. Start
docker-compose -f docker-compose.prod.yml up -d
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MW_API_BASE_URL` | MediaWiki API URL (e.g., `https://wiki.example.com/api.php`) |
| `MW_BOT_USERNAME` | Bot account for wiki edits |
| `MW_BOT_PASSWORD` | Bot password |
| `OPENAI_API_KEY` | OpenAI API key |
| `JWT_MW_TO_MCP_SECRET` | Shared with MWAssistant extension |
| `JWT_MCP_TO_MW_SECRET` | Shared with MWAssistant extension |
| `DATA_ROOT_PATH` | Tenant data directory (default: `/app/data`) |

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

Data is isolated per-tenant at `/app/data/{wiki_id}/`.

### MediaWiki LocalSettings.php

```php
$wgMWAssistantMCPBaseUrl = 'https://mcp.example.com';
$wgMWAssistantWikiId = 'my-wiki';  // Unique per wiki
$wgMWAssistantJWTMWToMCPSecret = getenv('JWT_MW_TO_MCP_SECRET');
$wgMWAssistantJWTMCPToMWSecret = getenv('JWT_MCP_TO_MW_SECRET');
```

## Monitoring

Check health:
```bash
curl http://localhost:8000/health
```

View logs:
```bash
docker logs mw-mcp-server -f
```

## Backup

Backup tenant data:
```bash
docker run --rm -v mcp_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/mcp-backup.tar.gz /data
```
