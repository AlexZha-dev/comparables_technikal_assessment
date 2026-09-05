"""Typed application configuration via pydantic-settings.

Pattern (per project convention used in other Comparables repos):

- One `BaseSettings` at the root — the only place that reads env / .env.
- Sub-models are plain `BaseModel` (not `BaseSettings`) — they hold validated
  fields and any cross-field derivations via `model_post_init`.
- `env_nested_delimiter="__"` maps `API__HOST=...` → `settings.api.host`.

Why sub-models are `BaseModel` and not `BaseSettings`:

    `BaseSettings` triggers Pydantic's nested-settings loader, which can
    recurse on `model_dump()` / `model_dump_json()` when the parent also
    derives from `BaseSettings`. Plain `BaseModel` is a clean leaf — no env
    parsing — and gets composited by the root via `Field(default_factory=…)`.

Usage:
    from comparables.core.config import get_settings
    s = get_settings()
    s.api.host              # namespaced (preferred)
    s.limits.max_llm_calls_per_run

Compute cross-field defaults in a `BaseModel`'s `model_post_init` instead of
declaring `@property` shims — it survives `model_dump` and JSON serialization
without surprises.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─── Sub-models (BaseModel) ──────────────────────────────────────────
class ApiConfig(BaseModel):
    """Process-level knobs: where the API binds, how verbose it is.

    Env keys (via root's `env_nested_delimiter="__"`):
        API__ENV, API__HOST, API__PORT, API__RELOAD, API__LOG_LEVEL, API__LOG_JSON
    """

    env: Literal["dev", "test", "prod"] = "dev"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    reload: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, v: object) -> object:
        return str(v).upper()


class PathsConfig(BaseModel):
    """All on-disk artifacts. Env keys: `PATHS__*`."""

    companies_json: Path = Path("./companies.json")
    sqlite: Path = Path("./data/companies.sqlite")
    bm25_pickle: Path = Path("./data/bm25.pkl")
    runs_dir: Path = Path("./runs")

    def model_post_init(self, __context: object) -> None:
        # Ensure parent directories exist so the API doesn't crash on cold start.
        for p in (self.companies_json, self.sqlite, self.bm25_pickle, self.runs_dir):
            if p.parent and not p.parent.exists():
                p.parent.mkdir(parents=True, exist_ok=True)


class LLMConfig(BaseModel):
    """Ollama (OpenAI-compatible) client config. Env keys: `LLM__*`."""

    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str = "ministral-3:3b"
    timeout_s: float = Field(default=60.0, gt=0)

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")


class WorkflowLimits(BaseModel):
    """Per-run budgets enforced by the LangGraph workflow. Env keys: `LIMITS__*`."""

    timeout_s: float = Field(default=60.0, gt=0)
    max_retrieval_iterations: int = Field(default=2, ge=1, le=2)
    max_revised_searches: int = Field(default=1, ge=0, le=1)
    max_candidates_per_iter: int = Field(default=100, ge=1, le=100)
    max_candidates_to_validate: int = Field(default=10, ge=1, le=10)
    max_final_results: int = Field(default=10, ge=1, le=10)
    max_llm_calls_per_run: int = Field(default=5, ge=1, le=5)


class ScoringWeights(BaseModel):
    """Weighted BM25 plus filter-match score. Env keys: `SCORING__*`."""

    w_bm25: float = Field(default=0.7, ge=0.0, le=1.0)
    w_filters: float = Field(default=0.3, ge=0.0, le=1.0)

    def model_post_init(self, __context: object) -> None:
        # Soft validation: warn on non-normalized weights but don't reject.
        import warnings

        if abs(self.w_bm25 + self.w_filters - 1.0) > 1e-3:
            warnings.warn(
                f"scoring: w_bm25 + w_filters = {self.w_bm25 + self.w_filters:.3f} != 1.0; "
                "rankings will be off-scale.",
                UserWarning,
                stacklevel=2,
            )


# ─── Root (BaseSettings) ─────────────────────────────────────────────
class Settings(BaseSettings):
    """Root configuration. One place reads env vars / .env file."""

    api: ApiConfig = Field(default_factory=ApiConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    limits: WorkflowLimits = Field(default_factory=WorkflowLimits)
    scoring: ScoringWeights = Field(default_factory=ScoringWeights)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    # ─── Convenience properties ────────────────────────────────────────
    # Plain `@property` reads the namespaced sub-model without triggering
    # Pydantic's serialization machinery (no recursion). Used by ad-hoc code
    # that prefers a flat accessor.
    @property
    def is_dev(self) -> bool:
        return self.api.env == "dev"

    @property
    def is_test(self) -> bool:
        return self.api.env == "test"

    @property
    def ollama_base_url_alive(self) -> str:
        """OpenAI-compat → Ollama native endpoint for readiness pings."""
        if self.llm.base_url.endswith("/v1"):
            return self.llm.base_url[: -len("/v1")] + "/api/tags"
        return self.llm.base_url.rstrip("/") + "/api/tags"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor. Use as a FastAPI dependency."""
    return Settings()


def reset_settings_cache() -> None:
    """For tests that need to re-read .env after monkey-patching."""
    get_settings.cache_clear()


__all__ = [
    "ApiConfig",
    "LLMConfig",
    "PathsConfig",
    "ScoringWeights",
    "Settings",
    "WorkflowLimits",
    "get_settings",
    "reset_settings_cache",
]
