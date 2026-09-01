"""Tool contract: `Tool` protocol, `ToolResult`, and the `@tool` decorator.

Tools are deterministic (no LLM calls) and async. They receive a `RunContext`
so they can log events and access injected services.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from comparables.core.context import RunContext


@dataclass
class ToolResult:
    """Outcome of a tool invocation."""

    ok: bool
    data: Any = None
    error: str | None = None
    duration_ms: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSpec:
    """JSON-Schema-style description of a tool — visible to the LLM."""

    name: str
    description: str
    input_schema: dict[str, Any]


class Tool(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]

    async def run(self, args: dict[str, Any], ctx: RunContext) -> ToolResult: ...


# ─── Decorator ─────────────────────────────────────────────────────────
_Func = Callable[[dict[str, Any], RunContext], Awaitable[ToolResult]]


def tool(
    *,
    name: str,
    description: str,
    input_schema: dict[str, Any],
) -> Callable[[_Func], _Func]:
    """Tag a function with ToolSpec metadata. The function itself stays a normal
    async callable; the registry inspects these attributes.
    """

    def deco(fn: _Func) -> _Func:
        fn._tool_spec = ToolSpec(  # type: ignore[attr-defined]
            name=name, description=description, input_schema=input_schema
        )
        return fn

    return deco


# ─── Default timing wrapper ────────────────────────────────────────────
async def timed(fn: _Func, args: dict[str, Any], ctx: RunContext) -> ToolResult:
    start = time.perf_counter()
    try:
        res = await fn(args, ctx)
    except Exception as exc:  # never let a tool crash the workflow
        return ToolResult(
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
    if res.duration_ms == 0:
        res.duration_ms = int((time.perf_counter() - start) * 1000)
    return res


__all__ = ["Tool", "ToolSpec", "ToolResult", "tool", "timed"]
