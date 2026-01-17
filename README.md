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

- Python 3.11+
- PostgreSQL 16 with pgvector extension
- Docker (recommended)
- OpenAI API key

## Configuration

See `.env.example` for all configuration options.

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `WIKI_CREDS` | Yes | JSON map of wiki credentials (mw_to_mcp_secret, mcp_to_mw_secret) for each wiki ID |
| `DB_PASSWORD` | Yes | PostgreSQL password |
| `ADMIN_API_KEY` | Optional | API key for accessing the Usage Stats Dashboard |


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
| `/stats/usage` | GET | Logic-level usage stats (Admin only) |
| `/stats/dashboard` | GET | HTML Usage Dashboard (Admin only) |
| `/health` | GET | Health check |

## Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev,test]"

# Start PostgreSQL
docker-compose up postgres -d

# Run server
uvicorn mw_mcp_server.main:app --reload

# Run tests
pytest
```

## Documentation

- [Deployment Guide](docs/DEPLOYMENT.md)
- [Extension Integration](docs/extension_integration.md)

## License

MIT
