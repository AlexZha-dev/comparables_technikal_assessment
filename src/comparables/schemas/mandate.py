"""Pydantic DTOs for parsed mandate and search plan."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


# ─── Filter spec ───────────────────────────────────────────────────────
class FilterSpec(BaseModel):
    """Structured filters derived from the natural-language query."""

    industries: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    employee_min: int | None = None
    employee_max: int | None = None
    revenue_buckets: list[str] = Field(default_factory=list)
    founded_after: int | None = None
    founded_before: int | None = None
    keywords: list[str] = Field(default_factory=list)

    @field_validator("industries", "locations", mode="before")
    @classmethod
    def _coerce_none(cls, v):
        return [] if v is None else v


# ─── Parsed mandate (LLM output of parse_mandate) ────────────────────
class ParsedMandate(BaseModel):
    """Result of LLM parsing a natural-language query into a structured mandate."""

    intent: str = Field(description="Short restatement of the user's goal.")
    filters: FilterSpec = Field(default_factory=FilterSpec)
    must_haves: list[str] = Field(
        default_factory=list,
        description="Hard boolean constraints ('in Nordics', 'B2B', 'post-2020').",
    )


# ─── Search plan (LLM output of plan_search) ──────────────────────────
class SearchPlan(BaseModel):
    """Which tools to invoke and in what order."""

    use_bm25: bool = True
    use_filters: bool = True
    keyword_boost: float = Field(default=1.0, ge=0.0, le=5.0)
    limit_per_iter: int = Field(default=50, ge=1, le=100)
    rationale: str = Field(default="", description="Why this plan, in one line.")
