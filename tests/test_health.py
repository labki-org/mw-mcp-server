"""
Health Endpoint Tests

Tests for the /health and /health/ready endpoints.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from mw_mcp_server.api.health_routes import router, health


def test_health_liveness():
    """Basic /health liveness endpoint returns ok."""
    result = health()
    assert result == {"status": "ok"}
