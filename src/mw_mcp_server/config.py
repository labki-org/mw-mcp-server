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
from pydantic import Field, SecretStr, field_validator, model_validator
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
        
        # 1. Try to load from WIKI_CREDS env var
        creds_map = {}
        if isinstance(v, str) and v.strip():
            try:
                raw_map = json.loads(v)
                for wiki_id, creds in raw_map.items():
                    creds_map[wiki_id] = WikiCredentials(**creds)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in WIKI_CREDS: {exc}")
        elif isinstance(v, dict):
             creds_map = {k: WikiCredentials(**val) if isinstance(val, dict) else val for k, val in v.items()}

        # 2. If no WIKI_CREDS, try legacy env vars (backward compatibility)
        # Note: We access values from info.data because validation runs after env loading
        if not creds_map:

            
            # If we have legacy secrets, we treat them as a "default" fallback 
            # that can be used if a token doesn't match any specific wiki_id,
            # or we can map them to a specific default ID.
            # For now, let's keep the legacy secrets in the main Settings object 
            # and use them as global fallbacks in the code.
            pass
            
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


class WikiCredentials(BaseSettings):
    """Credentials for a specific MediaWiki instance."""
    mw_to_mcp_secret: SecretStr
    mcp_to_mw_secret: SecretStr


# ----------------------------------------------------------------------
# Global Application Settings Singleton
# ----------------------------------------------------------------------

settings = Settings()
