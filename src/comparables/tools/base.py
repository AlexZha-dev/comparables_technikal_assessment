"""Tool contract: Pydantic everywhere.

- `ToolSpec`        — Pydantic model: name, description, JSON schema for inputs.
- `ToolResult`      — Pydantic model: outcome of an invocation.
- `@tool` decorator — accepts a Pydantic input model class and auto-generates
                       the JSON schema the LLM sees. Args are validated
                       against the model on every call — bad input becomes a
                       `ToolResult(ok=False, error="...")` and never crashes
                       the workflow.

Why Pydantic for tools: typed inputs surface mismatches at the decorator
boundary instead of leaking through to the tool body; JSON Schema generation
is a single line, so there's no parallel manual dict to drift from the model.
"""
from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from comparables.core.context import RunContext


# ─── Public models ──────────────────────────────────────────────────
class ToolSpec(BaseModel):
    """JSON-schema-style description of a tool — visible to the LLM."""

    name: str
    description: str
    # JSON Schema dict (generated from the tool's input_model via
    # `model_json_schema()`). Stored as dict for easy inspection/serialization.
    input_schema: dict[str, Any] = Field(default_factory=dict)
    # The Pydantic model class used for input validation (kept as a
    # reference; not directly serialized, used by the registry).
    input_model: type[BaseModel] | None = None


class ToolResult(BaseModel):
    """Outcome of a tool invocation."""

    ok: bool
    data: Any = None
    error: str | None = None
    duration_ms: int = 0
    meta: dict[str, Any] = Field(default_factory=dict)


# ─── Decorator ─────────────────────────────────────────────────────
# A tool is an async function taking (input_model_instance, RunContext) ->
# ToolResult. The decorator stores metadata + auto-generated schema.
_Func = Callable[[BaseModel, RunContext], Awaitable[ToolResult]]


def tool(
    *,
    name: str,
    description: str,
    input_model: type[BaseModel],
) -> Callable[[_Func], _Func]:
    """Tag a function with ToolSpec metadata.

    The function itself stays a normal async callable; the registry inspects
    the `_tool_spec` attribute attached by this decorator.
    """

    # Auto-generate the JSON schema once at decoration time.
    schema = input_model.model_json_schema()

    def deco(fn: _Func) -> _Func:
        fn._tool_spec = ToolSpec(  # type: ignore[attr-defined]
            name=name,
            description=description,
            input_schema=schema,
            input_model=input_model,
        )
        return fn

    return deco


# ─── Default timing / validation wrapper ────────────────────────────
class ToolInvocationError(RuntimeError):
    """Raised internally by `timed()` when input args fail Pydantic validation.

    Caught and converted into `ToolResult(ok=False)` so the workflow never
    crashes on bad tool input.
    """


async def timed(fn: _Func, args: dict[str, Any], ctx: RunContext) -> ToolResult:
    """Validate inputs through Pydantic, time the call, never crash.

    Steps:
      1. Pull `_tool_spec.input_model` from `fn`.
      2. `model_validate(args)` — fails → ToolResult(ok=False, error=...).
      3. Else: `await fn(model, ctx)` and stamp `duration_ms`.
    """
    start = time.perf_counter()
    spec: ToolSpec = fn._tool_spec  # type: ignore[attr-defined]
    input_model = spec.input_model
    if input_model is None:
        return ToolResult(
            ok=False,
            error="tool missing _tool_spec.input_model",
            duration_ms=int((time.perf_counter() - start) * 1000),
        )

    try:
        parsed = input_model.model_validate(args)
    except ValidationError as exc:
        return ToolResult(
            ok=False,
            error=f"input validation failed: {exc.errors()}",
            duration_ms=int((time.perf_counter() - start) * 1000),
        )

    try:
        res: ToolResult = await fn(parsed, ctx)
    except Exception as exc:  # never let a tool crash the workflow
        return ToolResult(
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
    if res.duration_ms == 0:
        res.duration_ms = int((time.perf_counter() - start) * 1000)
    return res


# ─── Schema dump helper (for /api/v1/tools/* introspection) ─────────
def specs_summary(specs: list[ToolSpec]) -> str:
    """One-line-per-spec summary; used in logs."""
    lines = []
    for s in specs:
        lines.append(f"{s.name}({json.dumps(s.input_schema)[:80]}…)")
    return "\n".join(lines)


__all__ = [
    "Tool",
    "ToolInvocationError",
    "ToolResult",
    "ToolSpec",
    "specs_summary",
    "timed",
    "tool",
]


# Backwards-compat alias so anything that imported the old Protocol keeps working.
from typing import Protocol  # noqa: E402


class Tool(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]

    async def run(  # type: ignore[override]
        self, args: dict[str, Any], ctx: RunContext
    ) -> ToolResult: ...
