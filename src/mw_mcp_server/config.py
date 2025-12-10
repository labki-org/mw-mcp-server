from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyHttpUrl

class Settings(BaseSettings):
    mw_api_base_url: AnyHttpUrl
    mw_bot_username: str
    mw_bot_password: str

    openai_api_key: str
    embedding_model: str = "text-embedding-3-large"

    # Bidirectional JWT secrets
    jwt_mw_to_mcp_secret: str  # For verifying tokens from MWAssistant
    jwt_mcp_to_mw_secret: str  # For signing tokens to MediaWiki
    
    # JWT constants (hardcoded as per spec)
    JWT_ALGO: str = "HS256"
    JWT_TTL: int = 30  # seconds

    vector_index_path: str = "/app/data/faiss_index.bin"
    vector_meta_path: str = "/app/data/index_meta.json"

    allowed_namespaces_public: str = "0,14"

    allowed_namespaces_public: str = "0,14"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

settings = Settings()


