"""Per-request run context: carries run_id, metrics, log buffer across nodes."""
from __future__ import annotations

import contextvars
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from comparables.core.exceptions import BudgetExceededError

# ContextVar so middleware → service → graph → tools all see the same run_id
run_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "run_id", default=None
)


@dataclass
class RunContext:
    """Mutable context shared across the workflow.

    Created per request by the API layer, passed through to nodes and tools.
    """

    run_id: str
    started_at: float = field(default_factory=time.time)
    llm_calls: int = 0
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    max_llm_calls: int = 5
    tool_calls: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    # Optional: injected services (set by app factory / DI)
    repos: Any = None
    llm: Any = None
    company_repo: Any = None
    bm25_repo: Any = None

    @staticmethod
    def new(*, max_llm_calls: int = 5) -> RunContext:
        """Create one context, reusing only an upstream request id.

        A locally generated id is deliberately not written back to the
        context variable. Direct service invocations therefore get a fresh id
        on every call, while HTTP requests still inherit the middleware id.
        """
        existing = run_id_var.get()
        rid = existing or uuid.uuid4().hex[:12]
        return RunContext(run_id=rid, max_llm_calls=max_llm_calls)

    def elapsed_ms(self) -> int:
        return int((time.time() - self.started_at) * 1000)

    def add_event(self, event_type: str, **payload: Any) -> None:
        """Append a structured event to the in-memory log buffer."""
        self.events.append(
            {
                "ts": time.time(),
                "type": event_type,
                **payload,
            }
        )

    def add_error(self, where: str, exc: Exception) -> None:
        self.errors.append(
            {
                "ts": time.time(),
                "where": where,
                "type": type(exc).__name__,
                "message": str(exc),
            }
        )

    def begin_llm_call(self) -> None:
        """Atomically reserve one provider call from the hard run budget."""
        if self.llm_calls >= self.max_llm_calls:
            raise BudgetExceededError(
                f"LLM call budget exhausted ({self.max_llm_calls} per run)"
            )
        self.llm_calls += 1

    def add_llm_usage(self, tokens_in: int = 0, tokens_out: int = 0) -> None:
        self.llm_tokens_in += max(0, tokens_in)
        self.llm_tokens_out += max(0, tokens_out)

    def inc_llm(self, tokens_in: int = 0, tokens_out: int = 0) -> None:
        """Compatibility helper for injected/fake LLM implementations."""
        self.begin_llm_call()
        self.add_llm_usage(tokens_in, tokens_out)

    def inc_tool(self) -> None:
        self.tool_calls += 1
