"""Tool: get_company — fetch a single record by id (for grounding)."""
from __future__ import annotations

from pydantic import BaseModel, Field

from comparables.core.context import RunContext
from comparables.tools.base import ToolResult, tool


# ─── Typed input ────────────────────────────────────────────────────
class GetCompanyInput(BaseModel):
    """Inputs for `get_company`."""

    company_id: int = Field(ge=1, description="Company id in the catalog (1..50_000).")


@tool(
    name="get_company",
    description=(
        "Fetch the full record for a single company by id. Use to ground evidence "
        "spans or to enrich a candidate with its full text before validation."
    ),
    input_model=GetCompanyInput,
)
async def get_company(args: GetCompanyInput, ctx: RunContext) -> ToolResult:
    company = getattr(ctx, "company_repo", None) or ctx.repos.company
    rec = await company.fetch_one(args.company_id)
    return ToolResult(ok=True, data=rec.model_dump())
