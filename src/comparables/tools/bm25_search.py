"""Tool: bm25_search — lexical search over company name+description."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from comparables.core.context import RunContext
from comparables.schemas.search import Hit  # typed result
from comparables.tools.base import ToolResult, tool


# ─── Typed input ────────────────────────────────────────────────────
class Bm25SearchInput(BaseModel):
    """Inputs for `bm25_search`."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        description="Free-text query (will be lowercased and tokenized).",
    )
    top_k: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Max results (1..100).",
    )
    candidate_ids: list[int] | None = Field(
        default=None,
        max_length=100,
        description="Optional eligibility pool to score without expanding retrieval.",
    )


@tool(
    name="bm25_search",
    description=(
        "Lexical (BM25) search over company name and description. "
        "Use for keyword-style queries like 'fraud detection', 'autonomous driving'. "
        "Returns up to top_k hits sorted by score desc."
    ),
    input_model=Bm25SearchInput,
)
async def bm25_search(args: Bm25SearchInput, ctx: RunContext) -> ToolResult:
    bm25 = getattr(ctx, "bm25_repo", None) or ctx.repos.bm25
    if args.candidate_ids is not None:
        hits: list[Hit] = await bm25.score_candidates(
            args.query,
            list(args.candidate_ids),
            top_k=args.top_k,
        )
    else:
        hits = await bm25.search(args.query, top_k=args.top_k)
    return ToolResult(ok=True, data=[h.model_dump() for h in hits])
