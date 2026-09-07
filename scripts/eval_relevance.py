"""Real retrieval and optional real-model semantic evaluation on labeled fixtures.

Run: python -m scripts.eval_relevance [--live] [--split holdout]
No fake/model-generated prediction is substituted when the provider is absent.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import pickle
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from comparables.agent.prompts import VALIDATE_CANDIDATE_SYSTEM
from comparables.core.config import get_settings
from comparables.core.context import RunContext
from comparables.db.base import Base
from comparables.db.session import Database
from comparables.ingestion.index import _build_bm25
from comparables.llm.client import LLMClient
from comparables.repositories.bm25_repo import BM25Repository
from comparables.repositories.company_repo import CompanyRepository
from comparables.services.filtered_search import FilteredSearchService
from comparables.services.semantic_validation import SemanticValidationService
from scripts.relevance_support import (
    RelevanceDataset,
    load_dataset,
    matches_filters,
    ranking_metrics,
    semantic_metrics,
)


async def evaluate_retrieval(
    dataset: RelevanceDataset, directory: Path, split: str
) -> list[dict[str, Any]]:
    records = dataset.records()
    db_path = directory / "relevance.sqlite"
    db_path.touch()
    database = Database(f"sqlite+aiosqlite:///{db_path.as_posix()}")
    await database.startup()
    try:
        async with database.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        catalog = CompanyRepository(database)
        await catalog.seed(list(records.values()))
        index, ids, lengths, avgdl = _build_bm25(list(records.values()))
        index_path = directory / "relevance.pkl"
        with index_path.open("wb") as handle:
            pickle.dump({"bm25": index, "ids": ids, "doc_lens": lengths, "avgdl": avgdl}, handle)
        ranker = BM25Repository(index_path)
        await ranker.load()
        service = FilteredSearchService(catalog, ranker)
        rows = []
        for case in dataset.retrieval_cases:
            if split != "all" and case.split != split:
                continue
            started = time.perf_counter()
            result = await service.search(case.query, case.filters, 100)
            ranked = [hit.company_id for hit in result.hits]
            contract_ok = (
                len(set(ranked)) == len(ranked) <= 100
                and all(
                    cid in records and matches_filters(records[cid], case.filters) for cid in ranked
                )
                and result.eligible_count
                == sum(matches_filters(record, case.filters) for record in records.values())
            )
            rows.append(
                {
                    "id": case.id,
                    "split": case.split,
                    "query": case.query,
                    "ranked_ids": ranked,
                    "relevant_ids": case.relevant_ids,
                    "eligible_count": result.eligible_count,
                    "contract_ok": contract_ok,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    **ranking_metrics(ranked, case.relevant_ids),
                }
            )
        return rows
    finally:
        await database.shutdown()


async def evaluate_semantics(
    dataset: RelevanceDataset, llm: LLMClient, split: str, timeout_s: float
) -> list[dict[str, Any]]:
    records = dataset.records()
    validator = SemanticValidationService(llm)
    rows = []
    for case in dataset.semantic_cases:
        if split != "all" and case.split != split:
            continue
        ctx = RunContext.new(max_llm_calls=2)
        started = time.perf_counter()
        requirements = dataset.requirements(case)
        try:
            async with asyncio.timeout(timeout_s):
                decisions, ok = await validator.validate(
                    case.query, [records[case.company_id]], requirements, ctx
                )
            decision = decisions[case.company_id]
        except TimeoutError as exc:
            from comparables.agent.semantic import SemanticAcceptancePolicy

            ctx.add_error("semantic_eval_timeout", exc)
            decision = SemanticAcceptancePolicy().assess(
                records[case.company_id], requirements, None
            )
            ok = False
        predicted = {item.criterion_id: item for item in decision.criteria}
        rows.append(
            {
                "id": case.id,
                "split": case.split,
                "tags": case.tags,
                "expected_relevant": case.relevant,
                "predicted_relevant": decision.relevant,
                "contract_ok": ok,
                "run_id": ctx.run_id,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "llm_calls": ctx.llm_calls,
                "tokens_in": ctx.llm_tokens_in,
                "tokens_out": ctx.llm_tokens_out,
                "estimated_cost_usd": None,
                "errors": ctx.errors,
                "criteria": [
                    {
                        "criterion_id": label.criterion_id,
                        "expected": label.status,
                        "predicted": predicted[label.criterion_id].status,
                        "evidence": [
                            item.model_dump() for item in predicted[label.criterion_id].evidence
                        ],
                    }
                    for label in case.labels
                ],
            }
        )
        print(
            f"[semantic] {case.id}: contract={ok}, expected={case.relevant}, accepted={decision.relevant}",
            flush=True,
        )
    return rows


def render_report(report: dict[str, Any]) -> str:
    def value(number):
        return "n/a" if number is None else f"{number:.3f}"

    lines = [
        "# Labeled relevance evaluation",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Dataset: `{report['dataset_sha256']}`; version {report['dataset_version']}.",
        f"Validator prompt SHA-256: `{report['prompt_sha256']}`.",
        f"Labels: {report['provenance']['label_author']}; human reviewed: {report['provenance']['human_reviewed']}.",
        "",
        report["provenance"]["purpose"],
        "",
        f"Semantic model: `{report['model']}`. Status: **{report['semantic_status']}**.",
        "",
        report["semantic_note"],
        "",
        "## Actual SQL/BM25 retrieval",
        "",
        "| Case | Split | Eligible | Retrieved | P@5 | Recall@100 | NDCG@5 | Contract |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["retrieval"]:
        lines.append(
            f"| {row['id']} | {row['split']} | {row['eligible_count']} | {len(row['ranked_ids'])} | {value(row['precision_at_5'])} | {value(row['recall_at_100'])} | {value(row['ndcg_at_5'])} | {'PASS' if row['contract_ok'] else 'FAIL'} |"
        )
    lines += [
        "",
        "P@5 has a fixed denominator of five, including unfilled slots. Recall/NDCG are undefined for no-positive queries. On this 15-record corpus recall@100 is easy: it does not prove production recall. Negated text can rank highly in lexical search. The separate 201-record regression test exercises the former early SQL LIMIT defect.",
        "",
        "## Real-model semantic decisions",
        "",
    ]
    if report["semantic_summary"] is None:
        lines.append(
            "Not measured. No semantic accuracy/precision score is inferred from contract tests or gold labels."
        )
    else:
        lines += [
            "| Split | Cases | Criterion accuracy | Precision | Recall | False accepts | Contract failures |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for split, metrics in report["semantic_summary"].items():
            lines.append(
                f"| {split} | {metrics['cases']} | {value(metrics['criterion_accuracy'])} | {value(metrics['acceptance_precision'])} | {value(metrics['acceptance_recall'])} | {metrics['false_accepts']} | {metrics['contract_failures']} |"
            )
    lines += [
        "",
        "## Scope and next evaluation",
        "",
        "The semantic suite isolates the real validator using pre-specified requirements; it does not measure parser completeness or end-to-end search relevance. Predictions use company records and requirements only; expected statuses and spans are never sent to the model. Contract failures remain in the metrics as fail-closed rejections, not skipped successes.",
        "",
        "The shared dev/holdout corpus is a regression split, not independent human ground truth. Before making business-quality claims: have two reviewers label unseen real catalog records, adjudicate disagreements, freeze a disjoint holdout, run the parser and whole workflow, and compare recall@100, precision@10, false accepts, latency and cost by prompt/model version. Exact quotes establish provenance, not entailment.",
        "",
        "Use `python -m scripts.eval_relevance --live --report eval/relevance-live.md` with the configured model available. A blocked provider returns exit code 2; contract failures or false accepts/rejects return 1. Offline mode tests retrieval only. Cost is unavailable unless provider pricing and usage can be verified.",
        "",
    ]
    return "\n".join(lines)


async def main(dataset_path: Path, report_path: Path, live: bool, split: str) -> int:
    dataset = load_dataset(dataset_path)
    settings = get_settings()
    with tempfile.TemporaryDirectory(prefix="comparables-relevance-") as directory:
        retrieval = await evaluate_retrieval(dataset, Path(directory), split)
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_version": dataset.version,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(VALIDATE_CANDIDATE_SYSTEM.encode()).hexdigest(),
        "provenance": dataset.provenance.model_dump(),
        "split": split,
        "model": settings.llm.model,
        "retrieval": retrieval,
        "semantic_status": "NOT_RUN",
        "semantic_summary": None,
        "semantic_rows": [],
        "semantic_note": "Offline mode: only retrieval and fixture integrity were evaluated.",
    }
    exit_code = 0 if all(row["contract_ok"] for row in retrieval) else 1
    if live:
        llm = LLMClient(
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
            model=settings.llm.model,
            timeout_s=settings.llm.timeout_s,
        )
        try:
            async with asyncio.timeout(min(10, settings.llm.timeout_s)):
                ready = await llm.ping()
        except TimeoutError:
            ready = False
        try:
            if not ready:
                report.update(
                    semantic_status="BLOCKED",
                    semantic_note="Configured model is unavailable at the configured endpoint. No semantic predictions were made.",
                )
                exit_code = 2
            else:
                rows = await evaluate_semantics(dataset, llm, split, settings.limits.timeout_s)
                summary = {
                    part: semantic_metrics([row for row in rows if row["split"] == part])
                    for part in ("dev", "holdout")
                    if any(row["split"] == part for row in rows)
                }
                report.update(
                    semantic_status="MEASURED",
                    semantic_rows=rows,
                    semantic_summary=summary,
                    semantic_note="Actual configured model; isolated validator evaluation against assistant-curated labels. Not independent human evaluation.",
                )
                if any(
                    item["contract_failures"]
                    or item["false_accepts"]
                    or item["false_rejects"]
                    or item["criterion_accuracy"] != 1
                    for item in summary.values()
                ):
                    exit_code = 1
        finally:
            await llm.aclose()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(report), encoding="utf-8")
    report_path.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[relevance] retrieval={len(retrieval)}; semantic={report['semantic_status']} -> {report_path}"
    )
    return exit_code


def cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("eval/relevance.json"))
    parser.add_argument("--report", type=Path, default=Path("eval/relevance-report.md"))
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--split", choices=["all", "dev", "holdout"], default="all")
    args = parser.parse_args()
    if (
        args.report.resolve() == args.dataset.resolve()
        or args.report.with_suffix(".json").resolve() == args.dataset.resolve()
    ):
        parser.error("report paths must not overwrite the dataset")
    return asyncio.run(main(args.dataset, args.report, args.live, args.split))


if __name__ == "__main__":
    raise SystemExit(cli())
