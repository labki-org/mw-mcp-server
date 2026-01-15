"""
Multi-Tenant Support

This module provides tenant isolation for serving multiple MediaWiki instances
from a single MCP server deployment.

Architecture
------------
- Each wiki is identified by a unique `wiki_id` (e.g., "wiki-alpha")
- JWTs from MWAssistant include a `wiki_id` claim
- Each wiki gets isolated storage under DATA_ROOT/{wiki_id}/
- Tenant context is extracted from authenticated requests

Security
--------
- wiki_id is validated to prevent path traversal attacks
- Only alphanumeric characters, hyphens, and underscores allowed
- Maximum 64 characters
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from .config import settings


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

WIKI_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


# ---------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------

class InvalidTenantError(ValueError):
    """Raised when a wiki_id is missing or malformed."""


# ---------------------------------------------------------------------
# Tenant Context Model
# ---------------------------------------------------------------------

class TenantContext(BaseModel):
    """
    Represents an isolated wiki tenant.

    This is typically derived from the `wiki_id` JWT claim.
    """

    wiki_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Unique identifier for the wiki instance.",
    )

    @field_validator("wiki_id", mode="before")
    @classmethod
    def validate_wiki_id(cls, v: str) -> str:
        """
        Validate wiki_id to prevent path traversal and injection.
        """
        if not v or not isinstance(v, str):
            raise InvalidTenantError("wiki_id is required")

        v = v.strip()

        if not WIKI_ID_PATTERN.match(v):
            raise InvalidTenantError(
                f"Invalid wiki_id '{v}': must be 1-64 alphanumeric chars, hyphens, or underscores"
            )

        # Extra safety: reject any path-like patterns
        if ".." in v or "/" in v or "\\" in v:
            raise InvalidTenantError(f"Invalid wiki_id '{v}': path traversal detected")

        return v


# ---------------------------------------------------------------------
# Tenant Path Utilities
# ---------------------------------------------------------------------

def get_tenant_data_root() -> Path:
    """
    Get the root directory for all tenant data.

    Defaults to /app/data or settings.data_root_path if configured.
    """
    root = getattr(settings, "data_root_path", None) or "/app/data"
    return Path(root)


def get_tenant_data_path(wiki_id: str) -> Path:
    """
    Get the data directory for a specific tenant.

    Parameters
    ----------
    wiki_id : str
        The validated wiki identifier.

    Returns
    -------
    Path
        Absolute path to the tenant's data directory.

    Raises
    ------
    InvalidTenantError
        If wiki_id is invalid.
    """
    # Validate through the model
    tenant = TenantContext(wiki_id=wiki_id)

    return get_tenant_data_root() / tenant.wiki_id


def get_tenant_index_path(wiki_id: str) -> str:
    """
    Get the FAISS index file path for a tenant.
    """
    return str(get_tenant_data_path(wiki_id) / "faiss_index.bin")


def get_tenant_meta_path(wiki_id: str) -> str:
    """
    Get the FAISS metadata file path for a tenant.
    """
    return str(get_tenant_data_path(wiki_id) / "index_meta.json")


def ensure_tenant_directory(wiki_id: str) -> Path:
    """
    Ensure the tenant's data directory exists.

    Returns the path to the created/existing directory.
    """
    path = get_tenant_data_path(wiki_id)
    path.mkdir(parents=True, exist_ok=True)
    return path
