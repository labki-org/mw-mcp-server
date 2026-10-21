# mw-mcp-server

Production-grade MCP (Model Context Protocol) server for MediaWiki + LLM integration.

## Features

- **Semantic Search**: Vector-based wiki content search using PostgreSQL + pgvector
- **Chat Interface**: Multi-turn conversational AI with persistent session history
- **MediaWiki Tools**: LLM-callable tools for page retrieval, SMW queries, and schema validation
- **Access Control**: JWT-based auth respecting MediaWiki permissions
- **Multi-Tenant**: Single server supports multiple wiki instances
- **Usage Stats**: Admin dashboard for tracking tokens, sessions, and active users

## Quick Start

```bash
# Clone and configure
git clone https://github.com/labki-org/mw-mcp-server.git
cd mw-mcp-server
cp .env.example .env

# Edit configuration
nano .env

# Start with Docker
docker-compose up -d
```

## Architecture

```
┌─────────────────────────────────────────────┐
│              mw-mcp-server                  │
│  ┌─────────────┐  ┌─────────────────────┐   │
│  │ Chat Routes │  │ Embedding Routes    │   │
│  │ /chat/*     │  │ /embeddings/*       │   │
│  └──────┬──────┘  └──────────┬──────────┘   │
│         │                    │              │
│  ┌──────▼────────────────────▼──────────┐   │
│  │         PostgreSQL + pgvector         │   │
│  │  • chat_session / chat_message       │   │
│  │  • embedding (vector similarity)     │   │
│  └───────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

## Requirements

- Python 3.11+ (the Docker image and CI use 3.12)
- PostgreSQL 16 with pgvector >= 0.7 (halfvec support; compose pins 0.8.7)
- Docker (recommended)
- OpenAI API key

## Configuration

See `.env.example` for all configuration options.

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `WIKI_CREDS` | Yes | JSON map of wiki ID -> `{mw_to_mcp_secret, mcp_to_mw_secret, api_url?}`. `api_url` is optional but recommended: it pins the wiki's `api.php` URL server-side instead of trusting the URL sent by the wiki (see [Deployment Guide](docs/DEPLOYMENT.md#multi-tenant-setup)) |
| `DB_PASSWORD` | Yes | PostgreSQL password (prod compose refuses to start without it) |
| `ADMIN_API_KEY` | Optional | API key for the Usage Stats Dashboard; sent in the `x-admin-key` header only |
| `ENABLE_API_DOCS` | Optional | `true` to serve `/docs`, `/redoc`, `/openapi.json` (default: disabled) |


## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat/` | POST | Chat completion with tools |
| `/chat/sessions` | GET | List user's chat sessions |
| `/chat/sessions/{id}` | GET | Get full session history |
| `/chat/sessions/{id}` | DELETE | Delete a session |
| `/search/` | POST | Vector-based semantic search |
| `/embeddings/page` | POST | Update page embedding |
| `/embeddings/page` | DELETE | Delete page embedding |
| `/embeddings/stats` | GET | Get embedding statistics |
| `/stats/usage` | GET | Usage stats (Admin only, `x-admin-key` header) |
| `/stats/dashboard` | GET | HTML Usage Dashboard (prompts for the admin key) |
| `/health` | GET | Liveness check |
| `/health/ready` | GET | Readiness check (database connectivity) |

## Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies (runtime lock + test/lint tooling)
pip install --require-hashes -r requirements.lock
pip install -r requirements-dev.txt
pip install --no-deps -e .

# Start PostgreSQL
docker-compose up postgres -d

# Run server
uvicorn mw_mcp_server.main:app --reload

# Run tests / lint / audit
pytest
ruff check src/ tests/
pip-audit -r requirements.lock --require-hashes --disable-pip
```

### Dependencies

- `requirements.txt` / `pyproject.toml`: runtime ranges (with security floors).
- `requirements.lock`: fully pinned + hashed set installed by the Docker image
  and CI. Regenerate after changing `requirements.txt`:
  `uv pip compile requirements.txt --universal --python-version 3.11 --generate-hashes -o requirements.lock`
- `requirements-dev.txt`: test/lint tooling, never shipped in the image.

## Documentation

- [Deployment Guide](docs/DEPLOYMENT.md)
- [Extension Integration](docs/extension_integration.md)

## License

MIT
