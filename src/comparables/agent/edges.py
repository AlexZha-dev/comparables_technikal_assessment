"""Conditional edge functions for the LangGraph workflow.

All hard limits live here so the graph topology is clean.
"""
from __future__ import annotations

from typing import Literal

from comparables.agent.state import AgentState


def after_parse(state: AgentState) -> Literal["plan_search", "finalize"]:
    if not state.get("parse_ok", False) and state.get("mandate") is None:
        return "finalize"
    return "plan_search"


def after_plan(state: AgentState) -> Literal["retrieve_and_score", "validate_candidates"]:
    # If we have a plan and it says no retrieval, skip directly to validate.
    plan = state.get("plan")
    if plan is not None and not plan.use_bm25 and not plan.use_filters:
        return "validate_candidates"
    return "retrieve_and_score"


def after_retrieve(
    state: AgentState,
    *,
    max_iterations: int,
    max_revised: int,
    revise_threshold: float = 0.05,
) -> Literal["revise_search", "validate_candidates"]:
    """Decide whether to revise or move to validation.

    Revise only if: iteration < max_iterations AND revised < max_revised
    AND current top score is below `revise_threshold` (i.e. retrieval is weak).
    """
    iter_n = int(state.get("iteration", 0))
    revised = bool(state.get("revised_search_done", False))
    cands = state.get("candidates") or []
    top_score = cands[0]["score"] if cands else 0.0

    if iter_n < max_iterations and not revised and top_score < revise_threshold:
        return "revise_search"
    return "validate_candidates"


def should_revise_again(state: AgentState) -> Literal["retrieve_and_score", "validate_candidates"]:
    return "retrieve_and_score"
