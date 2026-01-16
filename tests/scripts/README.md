# Manual Test Scripts

These scripts are for **manual integration testing** against a running server.
They are NOT meant to be run by pytest.

## Usage

1. Start the MCP server:
   ```bash
   docker compose up -d
   # Or: uvicorn mw_mcp_server.main:app --reload
   ```

2. Run manually:
   ```bash
   cd tests/scripts
   python manual_test_connection.py
   python manual_test_embed.py
   ```

## Scripts

- **manual_test_connection.py** - Tests chat endpoint, SMW queries, and session history
- **manual_test_embed.py** - Tests OpenAI embedding API connectivity  
- **verify_schema_tools.py** - Verification utility for schema tools

## Environment

These scripts read from `.env` in the project root for:
- `JWT_MW_TO_MCP_SECRET`
- `OPENAI_API_KEY`
