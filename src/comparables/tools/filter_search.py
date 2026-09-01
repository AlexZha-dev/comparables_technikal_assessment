"""Tool: filter_search — structured filters over the company catalog."""
from __future__ import annotations

from typing import Any

from comparables.core.context import RunContext
from comparables.schemas.mandate import FilterSpec
from comparables.tools.base import ToolResult, tool


@tool(
    name="filter_search",
    description=(
        "Find companies matching structured filters (industry, location, employee "
        "count range, revenue bucket, founded-year range). Use when the user gives "
        "explicit constraints. Returns up to top_k {company_id, score} pairs."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "industries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Allowed industry names (exact match).",
            },
            "locations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Allowed country/location names (exact match).",
            },
            "revenue_buckets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Allowed revenue-range buckets (e.g. '10M-50M').",
            },
            "employee_min": {"type": ["integer", "null"]},
            "employee_max": {"type": ["integer", "null"]},
            "founded_after": {"type": ["integer", "null"]},
            "founded_before": {"type": ["integer", "null"]},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 100, "default": 50},
        },
    },
)
async def filter_search(args: dict[str, Any], ctx: RunContext) -> ToolResult:
    f = FilterSpec(
        industries=list(args.get("industries") or []),
        locations=list(args.get("locations") or []),
        revenue_buckets=list(args.get("revenue_buckets") or []),
        employee_min=args.get("employee_min"),
        employee_max=args.get("employee_max"),
        founded_after=args.get("founded_after"),
        founded_before=args.get("founded_before"),
    )
    top_k: int = int(args.get("top_k", 50))
    company = getattr(ctx, "company_repo", None) or ctx.repos.company
    hits = await company.search_by_filters(f, top_k=top_k)
    return ToolResult(ok=True, data=[{"company_id": cid, "score": float(s)} for cid, s in hits])
