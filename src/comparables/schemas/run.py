"""Pydantic DTOs for run logs (observability)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RunEventType = Literal[
    "run_start",
    "parse_mandate",
    "plan_search",
    "tool_call",
    "retrieval_iter",
    "revise_search",
    "validation",
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
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    errors: list[dict[str, Any]] = Field(default_factory=list)
