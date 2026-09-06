"""Tool: filter_search — structured filters over the company catalog."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from comparables.core.context import RunContext
from comparables.schemas.mandate import FilterSpec
from comparables.schemas.search import Hit
from comparables.tools.base import ToolResult, tool


# ─── Typed input ────────────────────────────────────────────────────
class FilterSearchInput(BaseModel):
    """Inputs for `filter_search`. All fields are optional; an empty filter
    set still returns `top_k` rows from the catalog (deterministic order)."""

    model_config = ConfigDict(extra="forbid")

    industries: list[str] = Field(
        default_factory=list,
        description="Allowed industry names (exact match).",
    )
    locations: list[str] = Field(
        default_factory=list,
        description="Allowed country/location names (exact match).",
    )
    revenue_buckets: list[str] = Field(
        default_factory=list,
        description="Allowed revenue-range buckets (e.g. '10M-50M').",
    )
    employee_min: int | None = Field(default=None, description="Min employee count.")
    employee_max: int | None = Field(default=None, description="Max employee count.")
    founded_after: int | None = Field(default=None, description="Year >= this.")
    founded_before: int | None = Field(default=None, description="Year <= this.")
    top_k: int = Field(default=50, ge=1, le=100)


@tool(
    name="filter_search",
    description=(
        "Find companies matching structured filters (industry, location, employee "
        "count range, revenue bucket, founded-year range). Use when the user gives "
        "explicit constraints. Returns up to top_k hits."
    ),
    input_model=FilterSearchInput,
)
async def filter_search(args: FilterSearchInput, ctx: RunContext) -> ToolResult:
    company = getattr(ctx, "company_repo", None) or ctx.repos.company
    spec = FilterSpec(
        industries=list(args.industries),
        locations=list(args.locations),
        revenue_buckets=list(args.revenue_buckets),
        employee_min=args.employee_min,
        employee_max=args.employee_max,
        founded_after=args.founded_after,
        founded_before=args.founded_before,
    )
    hits: list[Hit] = await company.search_by_filters(spec, top_k=args.top_k)
    return ToolResult(ok=True, data=[h.model_dump() for h in hits])
