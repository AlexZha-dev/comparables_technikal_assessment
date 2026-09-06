"""Deterministic evaluation for the five assessment queries.

Parser correctness, hard-filter enforcement, catalog existence, and evidence
grounding are separate checks. Correct parsing cannot hide an invalid result.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from comparables.agent.semantic import semantic_criteria
from comparables.core.config import get_settings
from comparables.core.logging import configure_logging, get_logger
from comparables.db.session import Database
from comparables.llm.client import LLMClient
from comparables.repositories.bm25_repo import BM25Repository
from comparables.repositories.company_repo import CompanyRepository
from comparables.repositories.run_repo import RunRepository
from comparables.schemas.search import SemanticCriterion
from comparables.services.workflow_service import WorkflowService

logger = get_logger(__name__)


def _check(condition: bool, failure: str, success: str = "ok") -> tuple[bool, str]:
    return condition, success if condition else failure


def _parsed_criteria_match(actual: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, str]:
    issues: list[str] = []
    for field in ("industries", "locations", "revenue_buckets"):
        wanted = set(expected.get(field, []) or [])
        got = set(actual.get(field, []) or [])
        if wanted != got:
            issues.append(f"{field}: expected {sorted(wanted)}, got {sorted(got)}")
    for field in ("employee_min", "employee_max", "founded_after", "founded_before"):
        if actual.get(field) != expected.get(field):
            issues.append(f"{field}: expected {expected.get(field)}, got {actual.get(field)}")
    if expected.get("keywords_any"):
        keywords = " ".join(actual.get("keywords", []) or []).casefold()
        if not any(keyword.casefold() in keywords for keyword in expected["keywords_any"]):
            issues.append(
                f"keywords: expected any of {expected['keywords_any']}, "
                f"got {actual.get('keywords', [])}"
            )
    return not issues, "; ".join(issues) or "all expected criteria parsed"


def _company_matches_filters(company: dict[str, Any], filters: dict[str, Any]) -> bool:
    return not (
        (filters.get("industries") and company["industry"] not in filters["industries"])
        or (filters.get("locations") and company["location"] not in filters["locations"])
        or (
            filters.get("revenue_buckets")
            and company["revenue_range"] not in filters["revenue_buckets"]
        )
        or (
            filters.get("employee_min") is not None
            and company["employee_count"] < filters["employee_min"]
        )
        or (
            filters.get("employee_max") is not None
            and company["employee_count"] > filters["employee_max"]
        )
        or (
            filters.get("founded_after") is not None
            and company["founded_year"] < filters["founded_after"]
        )
        or (
            filters.get("founded_before") is not None
            and company["founded_year"] > filters["founded_before"]
        )
    )


def _mandatory_filters_applied(
    candidates: list[dict[str, Any]], filters: dict[str, Any]
) -> tuple[bool, str]:
    invalid = [
        candidate["company"]["id"]
        for candidate in candidates
        if not _company_matches_filters(candidate["company"], filters)
    ]
    return _check(
        not invalid,
        f"returned ids violating mandatory filters: {invalid}",
        f"all {len(candidates)} returned records satisfy every mandatory filter",
    )


def _evidence_grounded(candidates: list[dict[str, Any]]) -> tuple[bool, str]:
    for candidate in candidates:
        company = candidate["company"]
        if not candidate.get("relevant"):
            return False, f"id={company['id']} was returned with relevant=false"
        evidence = candidate.get("evidence", [])
        if not evidence:
            return False, f"id={company['id']} has no evidence"
        for item in evidence:
            field = item.get("field")
            if field not in company:
                return False, f"id={company['id']} references unknown field {field!r}"
            if not item.get("span") or item["span"] not in str(company[field]):
                return False, (
                    f"id={company['id']} span {item.get('span')!r} is not in {field}"
                )
    return True, f"all {len(candidates)} returned records have field-level evidence"


async def _ids_exist(repo: CompanyRepository, ids: list[int]) -> tuple[bool, str]:
    found = {record.id for record in await repo.fetch_by_ids(ids)}
    missing = sorted(set(ids) - found)
    return _check(not missing, f"ids absent from SQLite: {missing}", f"verified {len(ids)} ids")


def _semantic_contract(candidates: list[dict[str, Any]], criteria: list[SemanticCriterion]) -> tuple[bool, str]:
    expected = {criterion.criterion_id: criterion.requirement for criterion in criteria}
    for candidate in candidates:
        supplied = candidate.get("criteria", [])
        actual = {item["criterion_id"]: item["requirement"] for item in supplied}
        if actual != expected or len(supplied) != len(expected):
            return False, f"id={candidate['company']['id']}: incomplete semantic criteria"
        for item in supplied:
            if item["status"] != "supported" or not item.get("evidence"):
                return False, "returned candidate contains an unsupported requirement"
            for evidence in item["evidence"]:
                if evidence["field"] not in {"name", "description"} or not evidence["span"] or evidence["span"] not in candidate["company"][evidence["field"]]:
                    return False, "per-criterion evidence is untraceable"
    return True, "complete supported criterion verdicts; this checks the contract, not semantic truth"


async def _run_query(
    service: WorkflowService,
    company_repo: CompanyRepository,
    run_repo: RunRepository,
    query: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = await service.invoke(query["query"])
    except Exception as exc:
        run_id = getattr(exc, "run_id", None) or f"failed-{query['id']}"
        run_log = await run_repo.read(run_id) if getattr(exc, "run_id", None) else None
        wall_latency_ms = int((time.perf_counter() - started) * 1000)
        return {
            "id": query["id"],
            "query": query["query"],
            "passed": False,
            "checks": {
                "workflow_execution": {
                    "ok": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            },
            "metrics": {
                "latency_ms": run_log.latency_ms if run_log else wall_latency_ms,
                "wall_latency_ms": wall_latency_ms,
                "llm_calls": run_log.llm_calls if run_log else 0,
                "tool_calls": run_log.tool_calls if run_log else 0,
                "tokens_in": run_log.tokens_in if run_log else 0,
                "tokens_out": run_log.tokens_out if run_log else 0,
                "estimated_cost_usd": (
                    run_log.estimated_cost_usd if run_log else None
                ),
                "iterations": run_log.iterations if run_log else 0,
                "revised_searches": run_log.revised_searches if run_log else 0,
                "retrieved_count": run_log.retrieved_candidates if run_log else 0,
                "validated_count": run_log.validated_candidates if run_log else 0,
                "returned_count": run_log.final_count if run_log else 0,
                "mandate_filters": {},
            },
            "top_results": [],
            "run_id": run_id,
            "limitation": f"workflow failed: {type(exc).__name__}: {exc}",
        }
    wall_latency_ms = int((time.perf_counter() - started) * 1000)
    expected = query["expected"]
    candidates = [candidate.model_dump() for candidate in response.final]
    company_ids = [candidate["company"]["id"] for candidate in candidates]
    actual_filters = (
        response.mandate.filters.model_dump() if response.mandate is not None else {}
    )
    run_log = await run_repo.read(response.run_id)

    checks: dict[str, tuple[bool, str]] = {}
    checks["parsed_criteria"] = (
        _parsed_criteria_match(actual_filters, expected)
        if response.mandate is not None
        else (False, "mandate is null")
    )
    checks["mandatory_filters"] = _mandatory_filters_applied(candidates, actual_filters)
    checks["dataset_existence"] = await _ids_exist(company_repo, company_ids)
    source_records = {
        record.id: record.model_dump()
        for record in await company_repo.fetch_by_ids(company_ids)
    }
    checks["record_fidelity"] = _check(
        all(candidate["company"] == source_records.get(candidate["company"]["id"]) for candidate in candidates),
        "returned record differs from the source record in SQLite",
    )
    checks["unique_results"] = _check(
        len(company_ids) == len(set(company_ids)), "duplicate company IDs returned"
    )
    checks["evidence_traceability"] = _evidence_grounded(candidates)
    checks["semantic_validation_contract"] = _semantic_contract(
        candidates, semantic_criteria(response.mandate) if response.mandate else []
    )
    checks["candidate_pool_bound"] = _check(
        response.retrieved_candidates <= 100
        and run_log is not None
        and all(
            len(event.data.get("candidate_ids", []))
            == len(set(event.data.get("candidate_ids", [])))
            == event.data.get("candidates_after_cap", -1)
            <= 100
            for event in run_log.events if event.type == "retrieval_iter"
        ),
        "an iteration exceeds 100, contains duplicate IDs, or has missing pool telemetry",
    )
    checks["validation_bound"] = _check(
        response.validated_candidates <= 10,
        f"validated_candidates={response.validated_candidates} exceeds 10",
    )
    checks["result_bound"] = _check(len(candidates) <= 10, f"returned={len(candidates)} exceeds 10")
    checks["llm_budget"] = _check(response.llm_calls <= 5, f"llm_calls={response.llm_calls}")
    checks["iteration_bound"] = _check(response.iterations <= 2, f"iterations={response.iterations}")
    checks["revision_bound"] = _check(
        response.revised_searches <= 1
        and response.revised_search == (response.revised_searches == 1),
        f"revision fields disagree: bool={response.revised_search}, count={response.revised_searches}",
    )
    checks["run_observability"] = _check(
        run_log is not None
        and run_log.outcome.startswith("completed")
        and run_log.llm_calls == response.llm_calls
        and run_log.retrieved_candidates == response.retrieved_candidates
        and run_log.validated_candidates == response.validated_candidates
        and run_log.final_count == len(candidates)
        and run_log.tool_calls == response.tool_calls
        and run_log.tokens_in == response.tokens_in
        and run_log.tokens_out == response.tokens_out
        and run_log.iterations == response.iterations
        and run_log.revised_searches == response.revised_searches,
        "run_end metrics missing or inconsistent",
    )
    if not expected.get("allow_empty", False):
        minimum = expected.get("min_results", 1)
        checks["minimum_results"] = _check(
            len(candidates) >= minimum,
            f"returned={len(candidates)}, expected at least {minimum}",
        )

    passed = all(result for result, _ in checks.values())
    limitation = "none observed"
    if response.errors:
        limitation = f"workflow reported {len(response.errors)} recoverable error(s)"
    elif not candidates:
        limitation = "no company accepted; empty output alone does not prove zero eligible records in the full dataset"

    return {
        "id": query["id"],
        "query": query["query"],
        "passed": passed,
        "checks": {
            name: {"ok": result, "detail": detail}
            for name, (result, detail) in checks.items()
        },
        "metrics": {
            "latency_ms": response.latency_ms,
            "wall_latency_ms": wall_latency_ms,
            "llm_calls": response.llm_calls,
            "tool_calls": response.tool_calls,
            "tokens_in": response.tokens_in,
            "tokens_out": response.tokens_out,
            "estimated_cost_usd": response.estimated_cost_usd,
            "iterations": response.iterations,
            "revised_searches": response.revised_searches,
            "retrieved_count": response.retrieved_candidates,
            "validated_count": response.validated_candidates,
            "returned_count": len(candidates),
            "mandate_filters": actual_filters,
        },
        "top_results": candidates[:5],
        "run_id": response.run_id,
        "limitation": limitation,
    }


def _render_report(rows: list[dict[str, Any]], *, model: str) -> str:
    degraded = sum(row["limitation"].startswith("workflow reported") for row in rows)
    lines = [
        "# Evaluation Report",
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        f"Total: **{sum(row['passed'] for row in rows)}/{len(rows)} passed**",
        "",
        f"Configured model: `{model}`. Runs reporting recovered errors: **{degraded}/{len(rows)}**.",
        "",
        "Result scope: deterministic contract checks. Passing in fallback mode does not "
        "prove successful live-model execution or semantic relevance.",
        "",
        "## Summary",
        "",
        "| ID | Result | Latency ms | LLM | Tools | Retrieved | Validated | Returned | Revision | Cost USD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["metrics"]
        cost = metrics["estimated_cost_usd"]
        lines.append(
            f"| {row['id']} | {'PASS' if row['passed'] else 'FAIL'} | "
            f"{metrics['latency_ms']} | {metrics['llm_calls']} | {metrics['tool_calls']} | "
            f"{metrics['retrieved_count']} | {metrics['validated_count']} | "
            f"{metrics['returned_count']} | {metrics['revised_searches']} | "
            f"{'n/a' if cost is None else f'{cost:.6f}'} |"
        )

    lines.extend(["", "## Per-query details", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['id']}: {row['query']}",
                "",
                f"- Result: **{'PASS' if row['passed'] else 'FAIL'}**",
                f"- Run ID: `{row['run_id']}`",
                f"- Provider tokens in/out: {row['metrics']['tokens_in']}/{row['metrics']['tokens_out']}",
                "- Parsed filters: `"
                + json.dumps(row["metrics"]["mandate_filters"], ensure_ascii=False)
                + "`",
            ]
        )
        for name, result in row["checks"].items():
            lines.append(
                f"- {'PASS' if result['ok'] else 'FAIL'} `{name}`: {result['detail']}"
            )
        lines.append(f"- Important failure or limitation: {row['limitation']}")
        if row["top_results"]:
            lines.append("- Top records:")
            for candidate in row["top_results"]:
                company = candidate["company"]
                lines.append(
                    f"  - [{company['id']}] {company['name']} — {company['industry']}, "
                    f"{company['location']}; evidence={len(candidate['evidence'])}"
                )
        lines.append("")

    lines.extend(
        [
            "## Evaluation interpretation",
            "",
            "A successful result parses every explicit criterion, applies every structured "
            "constraint to every returned row, returns only catalog records, and provides "
            "non-empty field-level evidence while staying inside all run budgets.",
            "",
            "Parser fields, SQL eligibility, catalog existence, evidence substrings, counts, "
            "budgets, and telemetry are deterministic checks. Semantic relevance of free-text "
            "requirements still requires human review or a separately calibrated judge model.",
            "",
            "A production evaluation would add a versioned labeled query set, retrieval "
            "precision/recall and NDCG, parser field-level F1, adversarial and empty-result "
            "cases, regression thresholds by prompt/model version, sampled human review, and "
            "online latency/cost/false-positive dashboards.",
            "",
        ]
    )
    return "\n".join(lines)


async def main(report_path: Path) -> int:
    settings = get_settings()
    configure_logging(settings)
    database = Database.from_settings(settings)
    await database.startup()
    company_repo = CompanyRepository(database)
    await company_repo.connect()
    bm25_repo = BM25Repository(settings.paths.bm25_pickle)
    await bm25_repo.load()
    run_repo = RunRepository(settings.paths.runs_dir)
    llm = LLMClient(
        base_url=settings.llm.base_url,
        api_key=settings.llm.api_key,
        model=settings.llm.model,
        timeout_s=settings.llm.timeout_s,
    )
    service = WorkflowService(settings, company_repo, bm25_repo, llm, run_repo)

    try:
        queries = json.loads(Path("eval/queries.json").read_text(encoding="utf-8"))["queries"]
        rows: list[dict[str, Any]] = []
        seen_run_ids: set[str] = set()
        for query in queries:
            logger.info("eval.query", id=query["id"], query=query["query"])
            row = await _run_query(service, company_repo, run_repo, query)
            unique = row["run_id"] not in seen_run_ids
            row["checks"]["unique_run_id"] = {
                "ok": unique,
                "detail": "unique" if unique else "run id reused by another query",
            }
            row["passed"] = row["passed"] and unique
            seen_run_ids.add(row["run_id"])
            rows.append(row)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(_render_report(rows, model=settings.llm.model), encoding="utf-8")
        passed = sum(row["passed"] for row in rows)
        sys.stdout.write(f"\n[run_eval] {passed}/{len(rows)} passed -> {report_path}\n")
        return 0 if passed == len(rows) else 1
    finally:
        await llm.aclose()
        await database.shutdown()


def cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path("eval/report.md"))
    return asyncio.run(main(parser.parse_args().report))


if __name__ == "__main__":
    raise SystemExit(cli())
