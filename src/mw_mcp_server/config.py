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

from typing import List
from pydantic import Field, AnyHttpUrl, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Global application settings.

    All values are loaded from environment variables or a `.env` file.
    """

    # ------------------------------------------------------------------
    # MediaWiki Configuration
    # ------------------------------------------------------------------

    mw_api_base_url: AnyHttpUrl = Field(
        ...,
        description="Base URL of the MediaWiki API endpoint.",
    )

    mw_bot_username: str = Field(
        ...,
        min_length=1,
        description="Bot username for MediaWiki API authentication.",
    )

    mw_bot_password: SecretStr = Field(
        ...,
        min_length=1,
        description="Bot password for MediaWiki API authentication.",
    )

    # ------------------------------------------------------------------
    # OpenAI / LLM Configuration
    # ------------------------------------------------------------------

    openai_api_key: SecretStr = Field(
        ...,
        min_length=1,
        description="API key for OpenAI or OpenAI-compatible LLM provider.",
    )

    openai_model: str = Field(
        default="gpt-4o-mini",
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

    jwt_mw_to_mcp_secret: SecretStr = Field(
        ...,
        min_length=16,
        description="HMAC secret used to verify JWTs issued by MWAssistant.",
    )

    jwt_mcp_to_mw_secret: SecretStr = Field(
        ...,
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
        description="Maximum tokens a user can consume per day. Default: 100,000 (~$0.30/day at GPT-4o-mini prices).",
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


# ----------------------------------------------------------------------
# Global Application Settings Singleton
# ----------------------------------------------------------------------

settings = Settings()
