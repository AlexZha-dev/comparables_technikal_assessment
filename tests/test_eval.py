"""The evaluator must reject plausible-looking but incorrect responses."""
from comparables.schemas.search import SemanticCriterion
from scripts.run_eval import _evidence_grounded, _parsed_criteria_match, _semantic_contract


def test_parser_eval_rejects_broadened_location_set():
    ok, _ = _parsed_criteria_match(
        {"locations": ["Finland", "USA"]}, {"locations": ["Finland"]}
    )
    assert not ok


def test_parser_eval_rejects_invented_numeric_constraint():
    ok, _ = _parsed_criteria_match({"employee_min": 100}, {})
    assert not ok


def test_evidence_eval_rejects_empty_span():
    ok, _ = _evidence_grounded([
        {"company": {"id": 1, "name": "Example"}, "relevant": True,
         "evidence": [{"field": "name", "span": ""}]}
    ])
    assert not ok


def test_semantic_eval_rejects_omitted_requirement():
    candidate = {"company": {"id": 1}, "criteria": []}
    ok, _ = _semantic_contract([candidate], [SemanticCriterion(criterion_id="c1", requirement="uses AI")])
    assert not ok
