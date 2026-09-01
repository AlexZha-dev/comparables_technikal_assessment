"""Typed application configuration via pydantic-settings.

Reads from environment variables and `.env` file. Cached via `get_settings()`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings. Override anything via env var or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ─── App ────────────────────────────────────────────────────────────
    app_env: str = Field(default="dev", description="dev | prod | test")
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=True)

    # ─── Paths ──────────────────────────────────────────────────────────
    companies_json_path: Path = Field(default=Path("./companies.json"))
    sqlite_path: Path = Field(default=Path("./data/companies.sqlite"))
    bm25_pickle_path: Path = Field(default=Path("./data/bm25.pkl"))
    runs_dir: Path = Field(default=Path("./runs"))

    # ─── Ollama (OpenAI-compat) ─────────────────────────────────────────
    ollama_base_url: str = Field(default="http://localhost:11434/v1")
    ollama_api_key: str = Field(default="ollama")
    ollama_model: str = Field(default="qwen2.5:7b-instruct")
    ollama_timeout_s: float = Field(default=60.0)

    # ─── Workflow limits (per Comparables.ai assessment) ────────────────
    workflow_timeout_s: float = Field(default=60.0)
    max_retrieval_iterations: int = Field(default=2)
    max_revised_searches: int = Field(default=1)
    max_candidates_per_iter: int = Field(default=100)
    max_candidates_to_validate: int = Field(default=10)
    max_final_results: int = Field(default=10)
    max_llm_calls_per_run: int = Field(default=5)

    # ─── Retrieval scoring ──────────────────────────────────────────────
    score_w_bm25: float = Field(default=0.7)
    score_w_filters: float = Field(default=0.3)

    # ─── Derived flags ──────────────────────────────────────────────────
    @property
    def is_dev(self) -> bool:
        return self.app_env.lower() in {"dev", "development", "local"}

    @property
    def is_test(self) -> bool:
        return self.app_env.lower() in {"test", "testing"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor. Use as FastAPI dependency."""
    return Settings()
