"""The evaluation itself must detect bad labels and bad predictions."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.eval_relevance import evaluate_retrieval
from scripts.relevance_support import (
    RelevanceDataset,
    load_dataset,
    ranking_metrics,
    semantic_metrics,
)

DATASET = Path(__file__).resolve().parents[1] / "eval/relevance.json"


def test_labeled_fixture_provenance_and_requirements():
    dataset = load_dataset(DATASET)
    assert len(dataset.semantic_cases) == 24 and len(dataset.retrieval_cases) == 8
    assert not dataset.provenance.human_reviewed
    assert dataset.provenance.label_author == "assistant-curated"
    payload = [item.model_dump() for item in dataset.requirements(dataset.semantic_cases[0])]
    assert all(set(item) == {"criterion_id", "requirement"} for item in payload)


@pytest.mark.parametrize(
    "defect",
    [
        "span",
        "duplicate_company",
        "duplicate_case",
        "unknown_criterion",
        "filter_mismatch",
        "duplicate_label",
        "status",
    ],
)
def test_fixture_rejects_invalid_gold(defect):
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    label = data["semantic_cases"][0]["labels"][0]
    if defect == "span":
        label["spans"] = ["Invented quote"]
    elif defect == "duplicate_company":
        data["companies"].append(data["companies"][0])
    elif defect == "duplicate_case":
        data["semantic_cases"].append(data["semantic_cases"][0])
    elif defect == "unknown_criterion":
        label["criterion_id"] = "invented"
    elif defect == "filter_mismatch":
        data["retrieval_cases"][0]["relevant_ids"] = [14]
    elif defect == "duplicate_label":
        data["semantic_cases"][0]["labels"].append(label)
    else:
        label["status"] = "maybe"
    with pytest.raises(ValidationError):
        RelevanceDataset.model_validate(data)


def test_ranking_metrics_count_misses_and_rank_position():
    metrics = ranking_metrics([2, 1, 3], [1, 4])
    assert metrics["recall_at_100"] == 0.5
    assert metrics["precision_at_5"] == 0.2
    assert metrics["ndcg_at_5"] == pytest.approx(0.386852807)
    assert ranking_metrics([], [1])["recall_at_100"] == 0
    assert ranking_metrics([], [])["recall_at_100"] is None
    with pytest.raises(ValueError):
        ranking_metrics([1, 1], [1])


def test_fail_closed_predictions_do_not_claim_perfect_precision():
    rows = [
        {
            "predicted_relevant": False,
            "expected_relevant": expected,
            "contract_ok": False,
            "criteria": [
                {
                    "expected": "supported" if expected else "contradicted",
                    "predicted": "insufficient_evidence",
                }
            ],
        }
        for expected in [True, False]
    ]
    metrics = semantic_metrics(rows)
    assert metrics["acceptance_precision"] is None
    assert metrics["acceptance_recall"] == 0
    assert metrics["criterion_accuracy"] == 0
    assert metrics["false_rejects"] == 1 and metrics["contract_failures"] == 2


@pytest.mark.asyncio
async def test_real_sql_bm25_eval_on_fixture(tmp_path):
    rows = await evaluate_retrieval(load_dataset(DATASET), tmp_path, "all")
    assert len(rows) == 8 and all(row["contract_ok"] for row in rows)
    assert all(row["recall_at_100"] == 1 for row in rows if row["relevant_ids"])
    assert rows[-1]["ranked_ids"] == []
