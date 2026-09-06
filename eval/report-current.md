# Evaluation Report

_Generated: 2026-09-07 20:45:46_

Total: **1/5 passed**

Configured model: `ministral-3:3b`. Runs reporting recovered errors: **5/5**.

Result scope: deterministic contract checks. Passing in fallback mode does not prove successful live-model execution or semantic relevance.

## Summary

| ID | Result | Latency ms | LLM | Tools | Retrieved | Validated | Returned | Revision | Cost USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q1 | FAIL | 1553 | 2 | 1 | 100 | 10 | 0 | 0 | 0.000000 |
| q2 | FAIL | 89 | 2 | 1 | 100 | 10 | 0 | 0 | 0.000000 |
| q3 | FAIL | 370 | 3 | 2 | 100 | 10 | 0 | 1 | 0.000000 |
| q4 | FAIL | 110 | 2 | 1 | 100 | 10 | 0 | 0 | 0.000000 |
| q5 | PASS | 67 | 2 | 2 | 0 | 0 | 0 | 1 | 0.000000 |

## Per-query details

### q1: Find AI-driven fintech companies in the Nordics with more than 100 employees

- Result: **FAIL**
- Run ID: `ef508d67dba8`
- Provider tokens in/out: 0/0
- Parsed filters: `{"industries": ["Fintech"], "locations": ["Sweden", "Norway", "Finland"], "employee_min": 101, "employee_max": null, "revenue_buckets": [], "founded_after": null, "founded_before": null, "keywords": ["AI"]}`
- PASS `parsed_criteria`: all expected criteria parsed
- PASS `mandatory_filters`: all 0 returned records satisfy every mandatory filter
- PASS `dataset_existence`: verified 0 ids
- PASS `record_fidelity`: ok
- PASS `unique_results`: ok
- PASS `evidence_traceability`: all 0 returned records have field-level evidence
- PASS `semantic_validation_contract`: complete supported criterion verdicts; this checks the contract, not semantic truth
- PASS `candidate_pool_bound`: ok
- PASS `validation_bound`: ok
- PASS `result_bound`: ok
- PASS `llm_budget`: ok
- PASS `iteration_bound`: ok
- PASS `revision_bound`: ok
- PASS `run_observability`: ok
- FAIL `minimum_results`: returned=0, expected at least 1
- PASS `unique_run_id`: unique
- Important failure or limitation: workflow reported 2 recoverable error(s)

### q2: Renewable energy startups in Germany founded after 2018

- Result: **FAIL**
- Run ID: `7b9c9d65bbde`
- Provider tokens in/out: 0/0
- Parsed filters: `{"industries": ["Energy"], "locations": ["Germany"], "employee_min": null, "employee_max": null, "revenue_buckets": [], "founded_after": 2019, "founded_before": null, "keywords": ["renewable energy"]}`
- PASS `parsed_criteria`: all expected criteria parsed
- PASS `mandatory_filters`: all 0 returned records satisfy every mandatory filter
- PASS `dataset_existence`: verified 0 ids
- PASS `record_fidelity`: ok
- PASS `unique_results`: ok
- PASS `evidence_traceability`: all 0 returned records have field-level evidence
- PASS `semantic_validation_contract`: complete supported criterion verdicts; this checks the contract, not semantic truth
- PASS `candidate_pool_bound`: ok
- PASS `validation_bound`: ok
- PASS `result_bound`: ok
- PASS `llm_budget`: ok
- PASS `iteration_bound`: ok
- PASS `revision_bound`: ok
- PASS `run_observability`: ok
- FAIL `minimum_results`: returned=0, expected at least 1
- PASS `unique_run_id`: unique
- Important failure or limitation: workflow reported 2 recoverable error(s)

### q3: Healthcare companies with $50M-$100M revenue in the USA

- Result: **FAIL**
- Run ID: `bff1c03dc63c`
- Provider tokens in/out: 0/0
- Parsed filters: `{"industries": ["Healthcare"], "locations": ["USA"], "employee_min": null, "employee_max": null, "revenue_buckets": ["50M-100M"], "founded_after": null, "founded_before": null, "keywords": []}`
- PASS `parsed_criteria`: all expected criteria parsed
- PASS `mandatory_filters`: all 0 returned records satisfy every mandatory filter
- PASS `dataset_existence`: verified 0 ids
- PASS `record_fidelity`: ok
- PASS `unique_results`: ok
- PASS `evidence_traceability`: all 0 returned records have field-level evidence
- PASS `semantic_validation_contract`: complete supported criterion verdicts; this checks the contract, not semantic truth
- PASS `candidate_pool_bound`: ok
- PASS `validation_bound`: ok
- PASS `result_bound`: ok
- PASS `llm_budget`: ok
- PASS `iteration_bound`: ok
- PASS `revision_bound`: ok
- PASS `run_observability`: ok
- FAIL `minimum_results`: returned=0, expected at least 1
- PASS `unique_run_id`: unique
- Important failure or limitation: workflow reported 3 recoverable error(s)

### q4: Autonomous driving and machine learning companies in the Netherlands

- Result: **FAIL**
- Run ID: `3f02f77568b5`
- Provider tokens in/out: 0/0
- Parsed filters: `{"industries": ["Automotive"], "locations": ["Netherlands"], "employee_min": null, "employee_max": null, "revenue_buckets": [], "founded_after": null, "founded_before": null, "keywords": ["machine learning", "autonomous driving"]}`
- PASS `parsed_criteria`: all expected criteria parsed
- PASS `mandatory_filters`: all 0 returned records satisfy every mandatory filter
- PASS `dataset_existence`: verified 0 ids
- PASS `record_fidelity`: ok
- PASS `unique_results`: ok
- PASS `evidence_traceability`: all 0 returned records have field-level evidence
- PASS `semantic_validation_contract`: complete supported criterion verdicts; this checks the contract, not semantic truth
- PASS `candidate_pool_bound`: ok
- PASS `validation_bound`: ok
- PASS `result_bound`: ok
- PASS `llm_budget`: ok
- PASS `iteration_bound`: ok
- PASS `revision_bound`: ok
- PASS `run_observability`: ok
- FAIL `minimum_results`: returned=0, expected at least 1
- PASS `unique_run_id`: unique
- Important failure or limitation: workflow reported 2 recoverable error(s)

### q5: Biotech in France with more than 5000 employees

- Result: **PASS**
- Run ID: `94a670f37fc8`
- Provider tokens in/out: 0/0
- Parsed filters: `{"industries": ["Biotech"], "locations": ["France"], "employee_min": 5001, "employee_max": null, "revenue_buckets": [], "founded_after": null, "founded_before": null, "keywords": []}`
- PASS `parsed_criteria`: all expected criteria parsed
- PASS `mandatory_filters`: all 0 returned records satisfy every mandatory filter
- PASS `dataset_existence`: verified 0 ids
- PASS `record_fidelity`: ok
- PASS `unique_results`: ok
- PASS `evidence_traceability`: all 0 returned records have field-level evidence
- PASS `semantic_validation_contract`: complete supported criterion verdicts; this checks the contract, not semantic truth
- PASS `candidate_pool_bound`: ok
- PASS `validation_bound`: ok
- PASS `result_bound`: ok
- PASS `llm_budget`: ok
- PASS `iteration_bound`: ok
- PASS `revision_bound`: ok
- PASS `run_observability`: ok
- PASS `unique_run_id`: unique
- Important failure or limitation: workflow reported 2 recoverable error(s)

## Evaluation interpretation

A successful result parses every explicit criterion, applies every structured constraint to every returned row, returns only catalog records, and provides non-empty field-level evidence while staying inside all run budgets.

Parser fields, SQL eligibility, catalog existence, evidence substrings, counts, budgets, and telemetry are deterministic checks. Semantic relevance of free-text requirements still requires human review or a separately calibrated judge model.

A production evaluation would add a versioned labeled query set, retrieval precision/recall and NDCG, parser field-level F1, adversarial and empty-result cases, regression thresholds by prompt/model version, sampled human review, and online latency/cost/false-positive dashboards.
