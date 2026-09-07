# Labeled relevance evaluation

Generated: 2026-09-07T17:45:54.009007+00:00

Dataset: `d56138e478901800b8b8ba8f8b9e119ba6e628cb83dd306e1ff31ab1e5d75529`; version 1.0.
Validator prompt SHA-256: `abbce568c04f201260767bddbb3089973350f6f49a91e2ef6b2f7b30d243dab7`.
Labels: assistant-curated; human reviewed: False.

Transparent adversarial regression fixture, not independently human-labeled business ground truth. Dev/holdout is a maintenance split; the small shared corpus is not a statistical generalization benchmark.

Semantic model: `ministral-3:3b`. Status: **BLOCKED**.

Configured model is unavailable at the configured endpoint. No semantic predictions were made.

## Actual SQL/BM25 retrieval

| Case | Split | Eligible | Retrieved | P@5 | Recall@100 | NDCG@5 | Contract |
|---|---|---:|---:|---:|---:|---:|---|
| r01 | dev | 4 | 4 | 0.200 | 1.000 | 1.000 | PASS |
| r02 | dev | 3 | 3 | 0.200 | 1.000 | 0.500 | PASS |
| r03 | dev | 3 | 3 | 0.400 | 1.000 | 0.693 | PASS |
| r04 | dev | 2 | 2 | 0.200 | 1.000 | 1.000 | PASS |
| r05 | holdout | 2 | 2 | 0.200 | 1.000 | 0.631 | PASS |
| r06 | holdout | 1 | 1 | 0.200 | 1.000 | 1.000 | PASS |
| r07 | holdout | 3 | 3 | 0.200 | 1.000 | 1.000 | PASS |
| r08 | holdout | 0 | 0 | 0.000 | n/a | n/a | PASS |

P@5 has a fixed denominator of five, including unfilled slots. Recall/NDCG are undefined for no-positive queries. On this 15-record corpus recall@100 is easy: it does not prove production recall. Negated text can rank highly in lexical search. The separate 201-record regression test exercises the former early SQL LIMIT defect.

## Real-model semantic decisions

Not measured. No semantic accuracy/precision score is inferred from contract tests or gold labels.

## Scope and next evaluation

The semantic suite isolates the real validator using pre-specified requirements; it does not measure parser completeness or end-to-end search relevance. Predictions use company records and requirements only; expected statuses and spans are never sent to the model. Contract failures remain in the metrics as fail-closed rejections, not skipped successes.

The shared dev/holdout corpus is a regression split, not independent human ground truth. Before making business-quality claims: have two reviewers label unseen real catalog records, adjudicate disagreements, freeze a disjoint holdout, run the parser and whole workflow, and compare recall@100, precision@10, false accepts, latency and cost by prompt/model version. Exact quotes establish provenance, not entailment.

Use `python -m scripts.eval_relevance --live --report eval/relevance-live.md` with the configured model available. A blocked provider returns exit code 2; contract failures or false accepts/rejects return 1. Offline mode tests retrieval only. Cost is unavailable unless provider pricing and usage can be verified.
