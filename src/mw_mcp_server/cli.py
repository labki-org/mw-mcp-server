"""
CLI commands for mw-mcp-server maintenance tasks.

Usage:
    python -m mw_mcp_server.cli cleanup-sessions
"""

import asyncio
import logging
import sys

from .db import AsyncSessionLocal
from .db.cleanup import delete_expired_sessions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)


async def _cleanup_sessions() -> None:
    async with AsyncSessionLocal() as session:
        count = await delete_expired_sessions(session)
        print(f"Cleanup complete: {count} expired sessions deleted.")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m mw_mcp_server.cli <command>")
        print("Commands: cleanup-sessions")
        sys.exit(1)

    command = sys.argv[1]

    if command == "cleanup-sessions":
        asyncio.run(_cleanup_sessions())
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
