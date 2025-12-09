from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl

class Settings(BaseSettings):
    mw_api_base_url: AnyHttpUrl
    mw_bot_username: str
    mw_bot_password: str

    openai_api_key: str
    embedding_model: str = "text-embedding-3-large"

    jwt_secret: str
    jwt_algo: str = "HS256"

    vector_index_path: str = "/app/data/faiss_index.bin"
    vector_meta_path: str = "/app/data/index_meta.json"

    allowed_namespaces_public: str = "0,14"

    class Config:
        env_file = ".env"

settings = Settings()
