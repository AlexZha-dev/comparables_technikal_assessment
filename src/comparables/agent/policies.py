"""Deterministic domain policies used by the agent workflow.

The LLM interprets language and judges semantic relevance. Eligibility and
resource limits stay deterministic. Keeping these policies pure makes the
hard invariants independently testable and prevents prompt changes from
weakening mandatory filters.
"""
from __future__ import annotations

from dataclasses import dataclass

from comparables.agent.state import ScoredHit
from comparables.schemas.company import CompanyRecord
from comparables.schemas.mandate import FilterSpec
from comparables.schemas.search import Evidence


def has_structured_filters(filters: FilterSpec) -> bool:
    """Return whether at least one catalog-backed hard constraint is present."""
    return bool(
        filters.industries
        or filters.locations
        or filters.revenue_buckets
        or filters.employee_min is not None
        or filters.employee_max is not None
        or filters.founded_after is not None
        or filters.founded_before is not None
    )


@dataclass(frozen=True)
class EligibilityPolicy:
    """Specification for mandatory structured constraints."""

    filters: FilterSpec

    def matches(self, company: CompanyRecord) -> bool:
        f = self.filters
        return not (
            (f.industries and company.industry not in f.industries)
            or (f.locations and company.location not in f.locations)
            or (f.revenue_buckets and company.revenue_range not in f.revenue_buckets)
            or (f.employee_min is not None and company.employee_count < f.employee_min)
            or (f.employee_max is not None and company.employee_count > f.employee_max)
            or (f.founded_after is not None and company.founded_year < f.founded_after)
            or (f.founded_before is not None and company.founded_year > f.founded_before)
        )

    def evidence(self, company: CompanyRecord) -> list[Evidence]:
        """Produce traceable evidence for every structured constraint used."""
        if not self.matches(company):
            return []
        f = self.filters
        evidence: list[Evidence] = []
        if f.industries:
            evidence.append(Evidence(field="industry", span=company.industry))
        if f.locations:
            evidence.append(Evidence(field="location", span=company.location))
        if f.revenue_buckets:
            evidence.append(Evidence(field="revenue_range", span=company.revenue_range))
        if f.employee_min is not None or f.employee_max is not None:
            evidence.append(Evidence(field="employee_count", span=str(company.employee_count)))
        if f.founded_after is not None or f.founded_before is not None:
            evidence.append(Evidence(field="founded_year", span=str(company.founded_year)))
        return evidence


@dataclass(frozen=True)
class CandidateRanker:
    """Weighted ranking strategy with filter intersection as an invariant."""

    bm25_weight: float
    filter_weight: float

    def merge(
        self,
        bm25_hits: list[dict],
        filter_hits: list[dict],
        *,
        require_filter_match: bool,
        keyword_boost: float = 1.0,
    ) -> list[ScoredHit]:
        max_bm25 = max((float(hit["score"]) for hit in bm25_hits), default=1.0) or 1.0
        normalized_bm25 = {
            int(hit["company_id"]): float(hit["score"]) / max_bm25
            for hit in bm25_hits
        }
        filter_ids = {int(hit["company_id"]) for hit in filter_hits}
        eligible_ids = filter_ids if require_filter_match else set(normalized_bm25)

        ranked: list[ScoredHit] = []
        for company_id in eligible_ids:
            lexical = normalized_bm25.get(company_id, 0.0)
            filter_match = 1.0 if company_id in filter_ids else 0.0
            score = min(
                1.0,
                lexical * self.bm25_weight * keyword_boost
                + filter_match * self.filter_weight,
            )
            ranked.append(
                {
                    "company_id": company_id,
                    "bm25_score": lexical,
                    "filter_match": filter_match,
                    "score": score,
                }
            )
        return sorted(
            ranked,
            key=lambda hit: (-hit["score"], hit["company_id"]),
        )


__all__ = [
    "CandidateRanker",
    "EligibilityPolicy",
    "has_structured_filters",
]
