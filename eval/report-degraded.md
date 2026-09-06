# Evaluation Report

_Generated: 2026-09-07 15:44:30_

Total: **5/5 passed**

Configured model: `ministral-3:3b`. Runs reporting recovered errors: **5/5**.

Result scope: deterministic contract checks. Passing in fallback mode does not prove successful live-model execution or semantic relevance.

## Summary

| ID | Result | Latency ms | LLM | Tools | Retrieved | Validated | Returned | Revision | Cost USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q1 | PASS | 2646 | 2 | 2 | 50 | 10 | 7 | 0 | 0.000000 |
| q2 | PASS | 110 | 2 | 2 | 50 | 10 | 10 | 0 | 0.000000 |
| q3 | PASS | 57 | 1 | 1 | 50 | 10 | 10 | 0 | 0.000000 |
| q4 | PASS | 177 | 2 | 2 | 50 | 10 | 3 | 0 | 0.000000 |
| q5 | PASS | 67 | 1 | 1 | 0 | 0 | 0 | 0 | 0.000000 |

## Per-query details

### q1: Find AI-driven fintech companies in the Nordics with more than 100 employees

- Result: **PASS**
- Run ID: `f823a88cb418`
- Provider tokens in/out: 0/0
- Parsed filters: `{"industries": ["Fintech"], "locations": ["Sweden", "Norway", "Finland"], "employee_min": 101, "employee_max": null, "revenue_buckets": [], "founded_after": null, "founded_before": null, "keywords": ["AI"]}`
- PASS `parsed_criteria`: all expected criteria parsed
- PASS `mandatory_filters`: all 7 returned records satisfy every mandatory filter
- PASS `dataset_existence`: verified 7 ids
- PASS `record_fidelity`: ok
- PASS `unique_results`: ok
- PASS `evidence_traceability`: all 7 returned records have field-level evidence
- PASS `candidate_pool_bound`: ok
- PASS `validation_bound`: ok
- PASS `result_bound`: ok
- PASS `llm_budget`: ok
- PASS `iteration_bound`: ok
- PASS `revision_bound`: ok
- PASS `run_observability`: ok
- PASS `minimum_results`: ok
- PASS `unique_run_id`: unique
- Important failure or limitation: workflow reported 2 recoverable error(s)
- Top records:
  - [346] Davis PLC — Fintech, Finland; evidence=4
  - [476] Spencer-Garcia — Fintech, Finland; evidence=4
  - [919] Cruz-Allen — Fintech, Finland; evidence=4
  - [993] Orr Group — Fintech, Finland; evidence=4
  - [1235] Smith LLC — Fintech, Sweden; evidence=4

### q2: Renewable energy startups in Germany founded after 2018

- Result: **PASS**
- Run ID: `e8a2f1200a7e`
- Provider tokens in/out: 0/0
- Parsed filters: `{"industries": ["Energy"], "locations": ["Germany"], "employee_min": null, "employee_max": null, "revenue_buckets": [], "founded_after": 2019, "founded_before": null, "keywords": ["renewable energy"]}`
- PASS `parsed_criteria`: all expected criteria parsed
- PASS `mandatory_filters`: all 10 returned records satisfy every mandatory filter
- PASS `dataset_existence`: verified 10 ids
- PASS `record_fidelity`: ok
- PASS `unique_results`: ok
- PASS `evidence_traceability`: all 10 returned records have field-level evidence
- PASS `candidate_pool_bound`: ok
- PASS `validation_bound`: ok
- PASS `result_bound`: ok
- PASS `llm_budget`: ok
- PASS `iteration_bound`: ok
- PASS `revision_bound`: ok
- PASS `run_observability`: ok
- PASS `minimum_results`: ok
- PASS `unique_run_id`: unique
- Important failure or limitation: workflow reported 2 recoverable error(s)
- Top records:
  - [5038] Jones Group — Energy, Germany; evidence=4
  - [8411] Strickland-Wilson — Energy, Germany; evidence=4
  - [16713] Black-Salas — Energy, Germany; evidence=4
  - [17168] Webb-Ortega — Energy, Germany; evidence=4
  - [18823] Copeland-Morgan — Energy, Germany; evidence=4

### q3: Healthcare companies with $50M-$100M revenue in the USA

- Result: **PASS**
- Run ID: `9a3cea3ae732`
- Provider tokens in/out: 0/0
- Parsed filters: `{"industries": ["Healthcare"], "locations": ["USA"], "employee_min": null, "employee_max": null, "revenue_buckets": ["50M-100M"], "founded_after": null, "founded_before": null, "keywords": []}`
- PASS `parsed_criteria`: all expected criteria parsed
- PASS `mandatory_filters`: all 10 returned records satisfy every mandatory filter
- PASS `dataset_existence`: verified 10 ids
- PASS `record_fidelity`: ok
- PASS `unique_results`: ok
- PASS `evidence_traceability`: all 10 returned records have field-level evidence
- PASS `candidate_pool_bound`: ok
- PASS `validation_bound`: ok
- PASS `result_bound`: ok
- PASS `llm_budget`: ok
- PASS `iteration_bound`: ok
- PASS `revision_bound`: ok
- PASS `run_observability`: ok
- PASS `minimum_results`: ok
- PASS `unique_run_id`: unique
- Important failure or limitation: workflow reported 1 recoverable error(s)
- Top records:
  - [63] Brooks, Lam and Hayes — Healthcare, USA; evidence=3
  - [217] Miller Ltd — Healthcare, USA; evidence=3
  - [550] Jones, Compton and Day — Healthcare, USA; evidence=3
  - [922] Fisher, Payne and Thompson — Healthcare, USA; evidence=3
  - [1424] Gonzalez, Smith and Padilla — Healthcare, USA; evidence=3

### q4: Autonomous driving and machine learning companies in the Netherlands

- Result: **PASS**
- Run ID: `4dd2d93558d2`
- Provider tokens in/out: 0/0
- Parsed filters: `{"industries": ["Automotive"], "locations": ["Netherlands"], "employee_min": null, "employee_max": null, "revenue_buckets": [], "founded_after": null, "founded_before": null, "keywords": ["machine learning", "autonomous driving"]}`
- PASS `parsed_criteria`: all expected criteria parsed
- PASS `mandatory_filters`: all 3 returned records satisfy every mandatory filter
- PASS `dataset_existence`: verified 3 ids
- PASS `record_fidelity`: ok
- PASS `unique_results`: ok
- PASS `evidence_traceability`: all 3 returned records have field-level evidence
- PASS `candidate_pool_bound`: ok
- PASS `validation_bound`: ok
- PASS `result_bound`: ok
- PASS `llm_budget`: ok
- PASS `iteration_bound`: ok
- PASS `revision_bound`: ok
- PASS `run_observability`: ok
- PASS `minimum_results`: ok
- PASS `unique_run_id`: unique
- Important failure or limitation: workflow reported 2 recoverable error(s)
- Top records:
  - [1337] Hill-Ward — Automotive, Netherlands; evidence=4
  - [1578] Brown, Jensen and Rice — Automotive, Netherlands; evidence=4
  - [2402] Fisher, Perez and Pham — Automotive, Netherlands; evidence=4

### q5: Biotech in France with more than 5000 employees

- Result: **PASS**
- Run ID: `80460d829029`
- Provider tokens in/out: 0/0
- Parsed filters: `{"industries": ["Biotech"], "locations": ["France"], "employee_min": 5001, "employee_max": null, "revenue_buckets": [], "founded_after": null, "founded_before": null, "keywords": []}`
- PASS `parsed_criteria`: all expected criteria parsed
- PASS `mandatory_filters`: all 0 returned records satisfy every mandatory filter
- PASS `dataset_existence`: verified 0 ids
- PASS `record_fidelity`: ok
- PASS `unique_results`: ok
- PASS `evidence_traceability`: all 0 returned records have field-level evidence
- PASS `candidate_pool_bound`: ok
- PASS `validation_bound`: ok
- PASS `result_bound`: ok
- PASS `llm_budget`: ok
- PASS `iteration_bound`: ok
- PASS `revision_bound`: ok
- PASS `run_observability`: ok
- PASS `unique_run_id`: unique
- Important failure or limitation: workflow reported 1 recoverable error(s)

## Evaluation interpretation

A successful result parses every explicit criterion, applies every structured constraint to every returned row, returns only catalog records, and provides non-empty field-level evidence while staying inside all run budgets.

Parser fields, SQL eligibility, catalog existence, evidence substrings, counts, budgets, and telemetry are deterministic checks. Semantic relevance of free-text requirements still requires human review or a separately calibrated judge model.

A production evaluation would add a versioned labeled query set, retrieval precision/recall and NDCG, parser field-level F1, adversarial and empty-result cases, regression thresholds by prompt/model version, sampled human review, and online latency/cost/false-positive dashboards.
