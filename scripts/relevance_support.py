"""Versioned fixture contract and pure metrics, independent of the LLM adapter."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from comparables.schemas.company import CompanyRecord
from comparables.schemas.mandate import FilterSpec
from comparables.schemas.search import SemanticCriterion

Status = Literal["supported", "contradicted", "insufficient_evidence"]
Split = Literal["dev", "holdout"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Provenance(StrictModel):
    kind: str
    label_author: str
    human_reviewed: bool
    purpose: str


class Label(StrictModel):
    criterion_id: str
    status: Status
    spans: list[str]


class SemanticCase(StrictModel):
    id: str
    split: Split
    query: str
    company_id: int
    tags: list[str]
    labels: list[Label] = Field(min_length=1, max_length=12)

    @property
    def relevant(self) -> bool:
        return all(label.status == "supported" for label in self.labels)


class RetrievalCase(StrictModel):
    id: str
    split: Split
    query: str
    filters: FilterSpec
    relevant_ids: list[int]


class RelevanceDataset(StrictModel):
    version: str
    provenance: Provenance
    company_defaults: dict[str, Any]
    companies: list[dict[str, Any]] = Field(min_length=1)
    criteria: dict[str, str]
    semantic_cases: list[SemanticCase] = Field(min_length=1)
    retrieval_cases: list[RetrievalCase] = Field(min_length=1)

    def records(self) -> dict[int, CompanyRecord]:
        rows = [
            CompanyRecord.model_validate(self.company_defaults | company, strict=True)
            for company in self.companies
        ]
        if len({record.id for record in rows}) != len(rows):
            raise ValueError("duplicate company IDs")
        return {record.id: record for record in rows}

    def requirements(self, case: SemanticCase) -> list[SemanticCriterion]:
        # Only the requirement text/ID goes to the model; never the gold status/spans.
        return [
            SemanticCriterion(
                criterion_id=label.criterion_id, requirement=self.criteria[label.criterion_id]
            )
            for label in case.labels
        ]

    @model_validator(mode="after")
    def check_references(self) -> Self:
        records = self.records()
        cases = [*self.semantic_cases, *self.retrieval_cases]
        if len({case.id for case in cases}) != len(cases):
            raise ValueError("duplicate case IDs")
        if any(not key.strip() or not value.strip() for key, value in self.criteria.items()):
            raise ValueError("empty criterion")
        for case in self.semantic_cases:
            if case.company_id not in records:
                raise ValueError(f"{case.id}: unknown company")
            if len({label.criterion_id for label in case.labels}) != len(case.labels):
                raise ValueError(f"{case.id}: duplicate criterion")
            for label in case.labels:
                if label.criterion_id not in self.criteria:
                    raise ValueError(f"{case.id}: unknown criterion")
                if label.status != "insufficient_evidence" and not label.spans:
                    raise ValueError(f"{case.id}: decisive label needs evidence")
                if any(
                    not span or span not in records[case.company_id].description
                    for span in label.spans
                ):
                    raise ValueError(f"{case.id}: untraceable gold evidence")
        for case in self.retrieval_cases:
            if len(set(case.relevant_ids)) != len(case.relevant_ids):
                raise ValueError(f"{case.id}: duplicate relevant ID")
            for company_id in case.relevant_ids:
                if company_id not in records or not matches_filters(
                    records[company_id], case.filters
                ):
                    raise ValueError(f"{case.id}: gold ID absent or violates mandatory filters")
        return self


def load_dataset(path: Path) -> RelevanceDataset:
    return RelevanceDataset.model_validate_json(path.read_text(encoding="utf-8"))


def matches_filters(record: CompanyRecord, filters: FilterSpec) -> bool:
    return all(
        (
            not filters.industries or record.industry in filters.industries,
            not filters.locations or record.location in filters.locations,
            not filters.revenue_buckets or record.revenue_range in filters.revenue_buckets,
            filters.employee_min is None or record.employee_count >= filters.employee_min,
            filters.employee_max is None or record.employee_count <= filters.employee_max,
            filters.founded_after is None or record.founded_year >= filters.founded_after,
            filters.founded_before is None or record.founded_year <= filters.founded_before,
        )
    )


def ranking_metrics(ranked: list[int], relevant: list[int], k: int = 5) -> dict[str, float | None]:
    """Binary relevance. P@k uses fixed k (unfilled slots are not relevant).

    Recall/NDCG on no-positive queries are undefined, not perfect scores.
    Input is a unique ranked pool capped at 100, enforced by the runner.
    """
    if k <= 0 or len(set(ranked)) != len(ranked):
        raise ValueError("positive k and unique ranked IDs required")
    gold = set(relevant)
    dcg = sum(1 / math.log2(rank + 2) for rank, cid in enumerate(ranked[:k]) if cid in gold)
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(gold))))
    return {
        "precision_at_5": len(set(ranked[:k]) & gold) / k,
        "recall_at_100": len(set(ranked[:100]) & gold) / len(gold) if gold else None,
        "ndcg_at_5": dcg / ideal if ideal else None,
    }


def semantic_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(row["predicted_relevant"] and row["expected_relevant"] for row in rows)
    fp = sum(row["predicted_relevant"] and not row["expected_relevant"] for row in rows)
    fn = sum(not row["predicted_relevant"] and row["expected_relevant"] for row in rows)
    tn = len(rows) - tp - fp - fn
    pairs = [(label["expected"], label["predicted"]) for row in rows for label in row["criteria"]]
    statuses = ("supported", "contradicted", "insufficient_evidence")
    return {
        "cases": len(rows),
        "true_accepts": tp,
        "false_accepts": fp,
        "false_rejects": fn,
        "true_rejects": tn,
        "acceptance_precision": tp / (tp + fp) if tp + fp else None,
        "acceptance_recall": tp / (tp + fn) if tp + fn else None,
        "false_accept_rate": fp / (fp + tn) if fp + tn else None,
        "decision_accuracy": (tp + tn) / len(rows) if rows else None,
        "criterion_accuracy": sum(a == b for a, b in pairs) / len(pairs) if pairs else None,
        "contract_failures": sum(not row["contract_ok"] for row in rows),
        "confusion_matrix": {
            expected: {
                predicted: sum(a == expected and b == predicted for a, b in pairs)
                for predicted in statuses
            }
            for expected in statuses
        },
    }
