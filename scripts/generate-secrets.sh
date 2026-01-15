#!/bin/bash
# Generate secure secrets for mw-mcp-server deployment
#
# Usage:
#   ./scripts/generate-secrets.sh >> .env
#
# This generates the two JWT secrets needed for bidirectional
# authentication between MediaWiki and the MCP server.

set -euo pipefail

echo "# Generated secrets - $(date -Iseconds)"
echo "JWT_MW_TO_MCP_SECRET=$(openssl rand -base64 48)"
echo "JWT_MCP_TO_MW_SECRET=$(openssl rand -base64 48)"
