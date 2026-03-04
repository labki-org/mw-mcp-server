#!/bin/bash
# Generate secure secrets for mw-mcp-server deployment
#
# Usage:
#   ./scripts/generate-secrets.sh >> .env
#
# This generates the WIKI_CREDS JSON for bidirectional JWT authentication
# between MediaWiki and the MCP server. Replace "my-wiki" with your wiki_id.

set -euo pipefail

MW_TO_MCP=$(openssl rand -base64 48)
MCP_TO_MW=$(openssl rand -base64 48)

echo "# Generated secrets - $(date -Iseconds)"
echo "WIKI_CREDS='{\"my-wiki\": {\"mw_to_mcp_secret\": \"${MW_TO_MCP}\", \"mcp_to_mw_secret\": \"${MCP_TO_MW}\"}}'"
