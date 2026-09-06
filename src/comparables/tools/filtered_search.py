"""One bounded tool combining SQL eligibility and whole-pool lexical ranking."""

from pydantic import BaseModel, ConfigDict, Field

from comparables.core.context import RunContext
from comparables.schemas.mandate import FilterSpec
from comparables.services.filtered_search import FilteredSearchService
from comparables.tools.base import ToolResult, tool


class FilteredSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1)
    filters: FilterSpec
    top_k: int = Field(default=100, ge=1, le=100)


@tool(
    name="filtered_search",
    description="Rank all companies satisfying mandatory SQL filters by BM25, then return at most 100 hits. Eligibility IDs stay internal.",
    input_model=FilteredSearchInput,
)
async def filtered_search(args: FilteredSearchInput, ctx: RunContext) -> ToolResult:
    service = FilteredSearchService(ctx.repos.company, ctx.repos.bm25)
    result = await service.search(args.query, args.filters, args.top_k)
    return ToolResult(
        ok=True,
        data=[hit.model_dump() for hit in result.hits],
        meta={"eligible_count": result.eligible_count},
    )
