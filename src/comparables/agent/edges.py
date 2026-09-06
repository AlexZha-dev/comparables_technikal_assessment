"""Conditional edge functions for the LangGraph workflow.

All hard limits live here so the graph topology is clean.
"""
from __future__ import annotations

from typing import Literal

from comparables.agent.semantic import retrieval_query
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
    AND retrieval is empty or semantic keywords produced no lexical match.
    """
    iter_n = int(state.get("iteration", 0))
    revised = bool(state.get("revised_search_done", False))
    cands = state.get("candidates") or []
    mandate = state.get("mandate")
    keywords = retrieval_query(mandate) if mandate is not None else ""
    no_lexical_match = bool(keywords and cands) and all(
        float(candidate.get("bm25_score", 0.0)) <= 0.0 for candidate in cands
    )
    # With structured filters but no semantic terms there is nothing safe to
    # revise: hard filters are immutable, so an empty pool is a valid answer.
    weak_retrieval = bool(keywords) and (not cands or no_lexical_match)

    if (
        max_revised > 0
        and iter_n < max_iterations
        and not revised
        and weak_retrieval
    ):
        return "revise_search"
    return "validate_candidates"


def should_revise_again(state: AgentState) -> Literal["retrieve_and_score", "validate_candidates"]:
    return "retrieve_and_score"
