"""AgentState: shared state across all LangGraph nodes."""
from __future__ import annotations

from typing import Any, TypedDict

from comparables.schemas.company import CompanyRecord
from comparables.schemas.mandate import ParsedMandate, SearchPlan


class ScoredHit(TypedDict, total=False):
    """One retrieval hit, before/after scoring."""

    company_id: int
    bm25_score: float
    filter_match: float  # 0 or 1
    score: float  # combined


class ValidatedItem(TypedDict, total=False):
    company: CompanyRecord
    score: float
    relevant: bool
    evidence: list[dict[str, str]]
    reason: str


class AgentState(TypedDict, total=False):
    # ─── Identity / config ─────────────────────────────────────────────
    run_id: str
    raw_query: str
    settings_limits: dict[str, int]  # snapshot of settings.* at invoke time

    # ─── Parsing ───────────────────────────────────────────────────────
    mandate: ParsedMandate | None
    parse_ok: bool
    parse_error: str | None

    # ─── Planning ──────────────────────────────────────────────────────
    plan: SearchPlan | None
    plan_ok: bool

    # ─── Retrieval ─────────────────────────────────────────────────────
    candidates: list[ScoredHit]  # current accumulated candidates
    iteration: int               # 0..max_iterations
    revised_search_done: bool

    # ─── Validation ────────────────────────────────────────────────────
    validation: list[ValidatedItem]  # per-company decisions from LLM
    validation_ok: bool

    # ─── Output ────────────────────────────────────────────────────────
    final: list[ValidatedItem]
    errors: list[dict[str, Any]]
    llm_calls: int
    tokens_in: int
    tokens_out: int
    latency_ms: int
