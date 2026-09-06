"""ToolRegistry: name → tool lookup, invocation, OpenAPI-style spec dump."""
from __future__ import annotations

from typing import Any

from comparables.core.context import RunContext
from comparables.core.exceptions import RetrievalError
from comparables.tools.base import ToolResult, timed


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}  # name → callable with _tool_spec

    def register(self, fn: Any) -> None:
        spec = getattr(fn, "_tool_spec", None)
        if spec is None:
            raise ValueError(
                f"{fn!r} has no _tool_spec; wrap it with @tool(...)"
            )
        self._tools[spec.name] = fn

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        out = []
        for n in self.names():
            fn = self._tools[n]
            spec = fn._tool_spec
            out.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "input_schema": spec.input_schema,
                }
            )
        return out

    async def invoke(
        self, name: str, args: dict[str, Any], ctx: RunContext
    ) -> ToolResult:
        fn = self._tools.get(name)
        if fn is None:
            raise RetrievalError(f"Unknown tool: {name!r}; have {self.names()}")
        ctx.inc_tool()
        res = await timed(fn, args, ctx)
        if not res.ok:
            ctx.add_error(f"tool:{name}", RetrievalError(res.error or "tool failed"))
        ctx.add_event(
            "tool_call",
            tool=name,
            args=args,
            ok=res.ok,
            duration_ms=res.duration_ms,
            error=res.error,
            meta=res.meta,
        )
        return res


def default_registry() -> ToolRegistry:
    """Build the registry of bounded retrieval and hydration tools."""
    # Local import to avoid cycles
    from comparables.tools.bm25_search import bm25_search
    from comparables.tools.filter_search import filter_search
    from comparables.tools.filtered_search import filtered_search
    from comparables.tools.get_company import get_company

    reg = ToolRegistry()
    reg.register(bm25_search)
    reg.register(filter_search)
    reg.register(filtered_search)
    reg.register(get_company)
    return reg


__all__ = ["ToolRegistry", "default_registry"]
