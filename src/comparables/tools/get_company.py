"""Tool: get_company — fetch a single record by id (for grounding)."""
from __future__ import annotations

from typing import Any

from comparables.core.context import RunContext
from comparables.tools.base import ToolResult, tool


@tool(
    name="get_company",
    description=(
        "Fetch the full record for a single company by id. Use to ground evidence "
        "spans or to enrich a candidate with its full text before validation."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "company_id": {"type": "integer", "minimum": 1},
        },
        "required": ["company_id"],
    },
)
async def get_company(args: dict[str, Any], ctx: RunContext) -> ToolResult:
    cid: int = int(args["company_id"])
    company = getattr(ctx, "company_repo", None) or ctx.repos.company
    rec = await company.fetch_one(cid)
    return ToolResult(ok=True, data=rec.model_dump())
