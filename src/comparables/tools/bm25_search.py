"""Tool: bm25_search — lexical search over company name+description."""
from __future__ import annotations

from typing import Any

from comparables.core.context import RunContext
from comparables.tools.base import ToolResult, tool


@tool(
    name="bm25_search",
    description=(
        "Lexical (BM25) search over company name and description. "
        "Use for keyword-style queries like 'fraud detection', 'autonomous driving'. "
        "Returns up to top_k {company_id, score} pairs sorted by score desc."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Free-text query (will be lowercased and tokenized).",
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "default": 50,
                "description": "Max results (1..100).",
            },
        },
        "required": ["query"],
    },
)
async def bm25_search(args: dict[str, Any], ctx: RunContext) -> ToolResult:
    query: str = args.get("query", "").strip()
    top_k: int = int(args.get("top_k", 50))
    if not query:
        return ToolResult(ok=True, data=[])
    bm25 = getattr(ctx, "bm25_repo", None) or ctx.repos.bm25
    hits = await bm25.search(query, top_k=top_k)
    return ToolResult(ok=True, data=[{"company_id": cid, "score": float(s)} for cid, s in hits])
