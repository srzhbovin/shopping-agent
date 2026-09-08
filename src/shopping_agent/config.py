from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    catalog_db_path: Path = Path("data/catalog.db")
    demo_data_path: Path = Path("data/demo")
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "products_v1"
    embedding_base_url: str = "http://127.0.0.1:1234/v1"
    embedding_model: str = "text-embedding-nomic-embed-text-v1.5"
    embedding_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_base_url: str = "http://127.0.0.1:1234/api/v1"
    llm_model: str = "qwen/qwen3.5-4b"
    auto_load_demo: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
