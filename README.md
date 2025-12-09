# mw-mcp-server

A Python backend that exposes MediaWiki tools to LLMs via the Model Context Protocol (MCP). It allows LLMs to query content, run semantic queries, perform vector searches, and propose edits.

## Features

- **MCP Integration**: Exposes `get_page`, `run_smw_ask`, `vector_search` as tools.
- **HTTP API**:
    - `/chat`: Conversational endpoint.
    - `/search`: Vector search over wiki content.
    - `/smw-query`: Run Semantic MediaWiki `#ask` queries.
    - `/actions/edit-page`: Propose and apply edits.
- **Security**: JWT-based authentication for all protected endpoints.
- **Vector Search**: Local Faiss index for semantic retrieval.

## Setup

### Prerequisites
- Python 3.11+
- Docker (optional)

### Environment Variables
Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

| Variable | Description |
|----------|-------------|
| `MW_API_BASE_URL` | URL to your MediaWiki API (e.g., `https://wiki.local/api.php`) |
| `MW_BOT_USERNAME` | Bot username for edits |
| `MW_BOT_PASSWORD` | Bot password |
| `OPENAI_API_KEY` | Key for LLM and Embeddings |
| `JWT_SECRET` | Shared secret for verifying tokens from the extension |

### Running Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the server:
   ```bash
   uvicorn mw_mcp_server.main:app --reload
   ```

### Running with Docker

1. Build the image:
   ```bash
   docker build -t mw-mcp-server .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 --env-file .env mw-mcp-server
   ```

## Development
- **Source**: `src/mw_mcp_server`
- **Tests**: `tests/` (Run with `pytest`)
