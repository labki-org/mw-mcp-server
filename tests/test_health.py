"""
Health Endpoint Tests

Tests for the /health and /health/ready endpoints.
"""

from mw_mcp_server.api.health_routes import health


def test_health_liveness():
    """Basic /health liveness endpoint returns ok."""
    result = health()
    assert result == {"status": "ok"}
