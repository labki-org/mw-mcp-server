"""
Application Configuration

This module defines the complete, authoritative configuration schema for the
MCP server using Pydantic Settings.

Design Principles
-----------------
- Fail fast on misconfiguration (especially for secrets)
- Strong typing and strict validation
- Clear separation between:
    - External service credentials
    - JWT/security configuration
    - Database configuration
- Fully environment-driven (12-factor friendly)
- Test-friendly via direct instantiation overrides
"""

from __future__ import annotations

from typing import List, Dict, Optional
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Global application settings.

    All values are loaded from environment variables or a `.env` file.
    """

    # ------------------------------------------------------------------
    # MediaWiki Configuration
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # OpenAI / LLM Configuration
    # ------------------------------------------------------------------

    openai_api_key: SecretStr = Field(
        ...,
        min_length=1,
        description="API key for OpenAI or OpenAI-compatible LLM provider.",
    )

    openai_model: str = Field(
        default="gpt-4.1-mini",
        min_length=1,
        description="Default chat model for LLM completions.",
    )

    embedding_model: str = Field(
        default="text-embedding-3-large",
        min_length=1,
        description="Default model used for vector embeddings.",
    )

    embedding_dimensions: int = Field(
        default=3072,
        ge=256,
        le=4096,
        description="Dimensions of embedding vectors (must match model).",
    )

    # ------------------------------------------------------------------
    # JWT / Security Configuration (Bidirectional)
    # ------------------------------------------------------------------

    admin_api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key for accessing admin usage stats dashboard.",
    )

    jwt_mw_to_mcp_secret: Optional[SecretStr] = Field(
        default=None,
        min_length=16,
        description="HMAC secret used to verify JWTs issued by MWAssistant.",
    )

    jwt_mcp_to_mw_secret: Optional[SecretStr] = Field(
        default=None,
        min_length=16,
        description="HMAC secret used to sign JWTs sent to MediaWiki.",
    )

    jwt_algo: str = Field(
        default="HS256",
        description="JWT signing algorithm (must match both sides).",
    )

    jwt_ttl_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Lifetime of MCP→MW JWTs in seconds.",
    )

    # ------------------------------------------------------------------
    # Database Configuration (PostgreSQL + pgvector)
    # ------------------------------------------------------------------

    database_url: str = Field(
        default="postgresql+asyncpg://mcp:changeme@localhost:5432/mcp",
        min_length=1,
        description="SQLAlchemy async database URL for PostgreSQL.",
    )

    # ------------------------------------------------------------------
    # Rate Limiting Configuration
    # ------------------------------------------------------------------

    daily_token_limit: int = Field(
        default=100_000,
        ge=1000,
        description="Maximum tokens a user can consume per day. Default: 100,000 (~$0.80/day at GPT-4.1-mini prices).",
    )

    # ------------------------------------------------------------------
    # Session Retention
    # ------------------------------------------------------------------

    session_retention_days: int = Field(
        default=90,
        ge=1,
        le=3650,
        description="Number of days to retain chat sessions before archival/deletion.",
    )

    # ------------------------------------------------------------------
    # Multi-Tenant Data Storage
    # ------------------------------------------------------------------

    data_root_path: str = Field(
        default="/app/data",
        min_length=1,
        description="Root directory for tenant-scoped data.",
    )

    # ------------------------------------------------------------------
    # LLM Chat Loop Configuration
    # ------------------------------------------------------------------

    max_tool_loops: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of tool call iterations in the LLM chat loop.",
    )

    schema_cap: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum number of schema elements (categories/properties) to include in system prompt.",
    )

    # ------------------------------------------------------------------
    # Embedding Configuration
    # ------------------------------------------------------------------

    chunk_size: int = Field(
        default=12000,
        ge=500,
        le=100000,
        description="Text chunk size for embedding splits.",
    )

    chunk_overlap: int = Field(
        default=1200,
        ge=0,
        le=10000,
        description="Overlap between text chunks for embedding.",
    )

    # ------------------------------------------------------------------
    # Database Pool Configuration
    # ------------------------------------------------------------------

    db_pool_size: int = Field(
        default=5,
        ge=1,
        le=50,
        description="SQLAlchemy connection pool size.",
    )

    db_max_overflow: int = Field(
        default=10,
        ge=0,
        le=100,
        description="SQLAlchemy maximum connection pool overflow.",
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )

    embedding_queue_max_size: int = Field(
        default=1000,
        ge=10,
        le=100_000,
        description="Maximum number of pending embedding jobs in the queue.",
    )


    # ------------------------------------------------------------------
    # Namespace Access Control
    # ------------------------------------------------------------------

    allowed_namespaces_public: str = Field(
        default="0,14",
        description="Comma-separated list of MediaWiki namespace IDs allowed for public vector search.",
    )

    allowed_namespaces_public_list: List[int] = Field(
        default_factory=list,
        description="Parsed integer namespace IDs derived from allowed_namespaces_public.",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("allowed_namespaces_public_list", mode="before")
    @classmethod
    def _parse_allowed_namespaces(cls, v, info):
        """
        Parse comma-separated namespace string into integer list.
        """
        raw = info.data.get("allowed_namespaces_public", "")
        namespaces: List[int] = []

        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if not part.isdigit():
                raise ValueError(
                    f"Invalid namespace ID '{part}' in allowed_namespaces_public."
                )
            namespaces.append(int(part))

        return namespaces


    # ------------------------------------------------------------------
    # Pydantic Settings Configuration
    # ------------------------------------------------------------------

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",           # Allow extra env vars (like DB_PASSWORD)
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Multi-Wiki Credentials (JSON based)
    # ------------------------------------------------------------------
    
    wiki_creds: Dict[str, WikiCredentials] = Field(
        default_factory=dict,
        description="Map of wiki_id to credentials. Loaded from WIKI_CREDS JSON.",
    )

    @field_validator("wiki_creds", mode="before")
    @classmethod
    def _parse_wiki_creds(cls, v, info):
        """
        Parse WIKI_CREDS JSON string into dictionary.
        If empty, populates a default entry from legacy env vars if they exist.
        """
        import json

        creds_map = {}
        if isinstance(v, str) and v.strip():
            try:
                raw_map = json.loads(v)
                for wiki_id, creds in raw_map.items():
                    creds_map[wiki_id] = WikiCredentials(**creds)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in WIKI_CREDS: {exc}")
        elif isinstance(v, dict):
            creds_map = {
                k: WikiCredentials(**val) if isinstance(val, dict) else val
                for k, val in v.items()
            }

        return creds_map

    @model_validator(mode="after")
    def _validate_secrets_configuration(self) -> Settings:
        """
        Ensure that we have at least one valid set of credentials.
        """
        # If we have valid wiki_creds, we are good.
        if self.wiki_creds:
            return self

        # If no wiki_creds, we MUST have legacy secrets.
        if not self.jwt_mw_to_mcp_secret or not self.jwt_mcp_to_mw_secret:
            raise ValueError(
                "Configuration Error: You must provide either WIKI_CREDS (for multi-wiki) "
                "OR both jwt_mw_to_mcp_secret and jwt_mcp_to_mw_secret (legacy mode)."
            )
        
        return self


class WikiCredentials(BaseModel):
    """Credentials for a specific MediaWiki instance."""
    mw_to_mcp_secret: SecretStr
    mcp_to_mw_secret: SecretStr


# ----------------------------------------------------------------------
# Global Application Settings Singleton
# ----------------------------------------------------------------------

settings = Settings()
