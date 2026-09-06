"""Pydantic DTOs for the public /search API surface."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from comparables.schemas.company import CompanyRecord
from comparables.schemas.mandate import ParsedMandate


# ─── Retrieval primitives ───────────────────────────────────────────
class Hit(BaseModel):
    """One (company_id, score) pair produced by a retrieval tool.

    Lifted into a Pydantic model so the agent code (and tests) can pass
    typed hits instead of tuples / dicts. `score` semantics:
      - bm25_search: float, normalized later in the agent.
      - filter_search: always 1.0 (binary match).
    """

    model_config = ConfigDict(extra="forbid")

    company_id: int = Field(ge=1)
    score: float = Field(ge=0.0)


# ─── Public API types ───────────────────────────────────────────────
class Evidence(BaseModel):
    """Grounded support for a candidate: a literal substring of the record."""

    model_config = ConfigDict(extra="forbid")

    field: Literal[
        "name",
        "description",
        "industry",
        "location",
        "employee_count",
        "revenue_range",
        "founded_year",
    ] = Field(description="CompanyRecord field containing the exact span.")
    span: str = Field(min_length=1, description="Exact substring of `field`.")


class SemanticCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    criterion_id: str = Field(min_length=1)
    requirement: str = Field(min_length=1)


class SemanticEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: Literal["name", "description"]
    span: str = Field(min_length=1, max_length=1000)


class CriterionVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    criterion_id: str = Field(min_length=1)
    status: Literal["supported", "contradicted", "insufficient_evidence"]
    evidence: list[SemanticEvidence] = Field(default_factory=list, max_length=2)


class CriterionAssessment(CriterionVerdict):
    requirement: str


class Candidate(BaseModel):
    """One returned company with its grounded evidence and a 0..1 score."""

    company: CompanyRecord
    score: float = Field(ge=0.0, le=1.0)
    relevant: bool = True
    evidence: list[Evidence] = Field(default_factory=list)
    reason: str = ""
    criteria: list[CriterionAssessment] = Field(default_factory=list)


class CandidateVerdict(BaseModel):
    """Strict LLM output for one semantic relevance decision."""

    model_config = ConfigDict(extra="forbid")

    company_id: int = Field(ge=1)
    criteria: list[CriterionVerdict] = Field(min_length=1, max_length=12)


class BatchVerdicts(BaseModel):
    """Strict batched validator output, capped by the workflow at ten rows."""

    model_config = ConfigDict(extra="forbid")

    verdicts: list[CandidateVerdict] = Field(default_factory=list, max_length=10)


class SearchRequest(BaseModel):
    """Public request body for POST /api/v1/search."""

    query: str = Field(min_length=1, max_length=1000)


class SearchResponse(BaseModel):
    """Public response: the parsed mandate + final list of grounded candidates."""

    run_id: str
    query: str
    mandate: ParsedMandate | None = None
    final: list[Candidate] = Field(default_factory=list)
    llm_calls: int = 0
    tool_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    estimated_cost_usd: float | None = None
    retrieved_candidates: int = 0
    validated_candidates: int = 0
    revised_search: bool = False
    revised_searches: int = 0
    iterations: int = 0
    latency_ms: int = 0
    errors: list[dict] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")
