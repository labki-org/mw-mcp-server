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

from typing import Dict, Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class WikiCredentials(BaseModel):
    """Credentials (and optional pinned API URL) for a specific MediaWiki instance."""
    mw_to_mcp_secret: SecretStr
    mcp_to_mw_secret: SecretStr
    api_url: Optional[str] = Field(
        default=None,
        description=(
            "MediaWiki api.php URL for this wiki. When set, the server always "
            "calls this URL and rejects user tokens whose 'api_url' claim "
            "differs. When unset, the URL from the token is used (legacy)."
        ),
    )

    @field_validator("api_url")
    @classmethod
    def _validate_api_url(cls, v: Optional[str]) -> Optional[str]:
        if v is None or not v.strip():
            return None
        v = v.strip()
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("api_url must be an absolute http(s) URL.")
        return v


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

    jwt_algo: Literal["HS256"] = Field(
        default="HS256",
        description="JWT signing algorithm (must match both sides; MWAssistant only accepts HS256).",
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
        default=300,
        ge=10,
        le=2000,
        description=(
            "Maximum number of schema elements (categories/properties) to include "
            "in the system prompt. 300 fits comfortably in tokens and gives the "
            "LLM enough vocabulary to recognize concept synonyms before falling "
            "back to vector search."
        ),
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
        description="Maximum number of pending embedding jobs per wiki.",
    )

    embedding_queue_global_max_size: Optional[int] = Field(
        default=None,
        description="Cap on pending embedding jobs across all wikis (default: 10x embedding_queue_max_size).",
    )

    embedding_job_max_attempts: int = Field(
        default=6,
        ge=1, le=50,
        description="Attempts before a failing embedding job is dropped.",
    )

    embedding_job_backoff_base_seconds: float = Field(
        default=5,
        ge=0,
        description="Base delay for embedding job retry backoff.",
    )

    embedding_job_backoff_max_seconds: float = Field(
        default=900,
        ge=0,
        description="Maximum delay between embedding job retries.",
    )

    embedding_job_lease_seconds: float = Field(
        default=600,
        ge=10,
        description="How long a worker holds a claimed embedding job before it can be reclaimed.",
    )

    embedding_worker_poll_seconds: float = Field(
        default=5,
        gt=0,
        description="Embedding worker poll interval when the queue is empty.",
    )

    llm_max_retries: int = Field(
        default=3,
        ge=0, le=10,
        description="Retries for LLM calls on 408/409/429/5xx and timeouts.",
    )

    chat_context_token_budget: int = Field(
        default=60000,
        ge=1000,
        description="Token budget for the messages sent to the LLM per call; older history is trimmed to fit.",
    )

    tool_result_max_chars: int = Field(
        default=24000,
        ge=1000,
        description="Maximum characters of a single tool result passed to the LLM.",
    )

    chat_token_reservation: int = Field(
        default=8000,
        ge=0,
        description="Tokens reserved against the daily limit when a chat run starts (reconciled afterwards).",
    )

    chat_persist_timeout_seconds: float = Field(
        default=15,
        gt=0,
        description="Timeout for saving chat messages and usage after a run.",
    )

    session_cleanup_interval_seconds: float = Field(
        default=21600,
        gt=0,
        description="Interval between expired chat session cleanups.",
    )

    enable_api_docs: bool = Field(
        default=False,
        description="Serve /docs, /redoc and /openapi.json (ENABLE_API_DOCS).",
    )


    # ------------------------------------------------------------------
    # Namespace Access Control
    # ------------------------------------------------------------------

    allowed_namespaces_public: str = Field(
        default="0,14",
        description=(
            "Comma-separated list of MediaWiki namespace IDs allowed for public "
            "vector search. Currently unused: namespace access comes from the "
            "per-user 'allowed_namespaces' JWT claim."
        ),
    )


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



# ----------------------------------------------------------------------
# Global Application Settings Singleton
# ----------------------------------------------------------------------

settings = Settings()
