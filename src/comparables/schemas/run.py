"""Pydantic DTOs for run logs (observability)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RunEventType = Literal[
    "run_start",
    "parse_mandate",
    "plan_search",
    "llm_call",
    "tool_call",
    "retrieval_iter",
    "revise_search",
    "validation",
    "finalize",
    "run_end",
]


class RunEvent(BaseModel):
    """One line in runs/<run_id>.jsonl."""

    ts: float
    type: RunEventType
    data: dict[str, Any] = Field(default_factory=dict)


class RunLog(BaseModel):
    """Full log for a run, returned by GET /api/v1/runs/{run_id}."""

    run_id: str
    query: str
    events: list[RunEvent] = Field(default_factory=list)
    final_count: int = 0
    outcome: str = "unknown"
    llm_calls: int = 0
    tool_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    estimated_cost_usd: float | None = None
    retrieved_candidates: int = 0
    validated_candidates: int = 0
    iterations: int = 0
    revised_searches: int = 0
    latency_ms: int = 0
    errors: list[dict[str, Any]] = Field(default_factory=list)
