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
    llm_timeout_seconds: float = Field(default=55.0, gt=0)
    llm_max_output_tokens: int = Field(default=128, ge=32, le=1024)
    llm_max_retries: int = Field(default=1, ge=0, le=3)
    prompt_path: Path = Path("prompts")
    agent_max_steps: int = Field(default=12, ge=4, le=30)
    agent_max_tool_calls: int = Field(default=30, ge=1, le=100)
    agent_token_budget: int = Field(default=1200, ge=64, le=100000)
    agent_timeout_seconds: float = Field(default=90.0, gt=1)
    agent_max_critic_revisions: int = Field(default=2, ge=0, le=2)
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3000"
    auto_load_demo: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
