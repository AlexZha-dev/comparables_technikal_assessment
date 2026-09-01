"""RunRepository — JSONL append-only log per run, plus full read."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from comparables.core.exceptions import IndexNotFoundError
from comparables.schemas.run import RunEvent, RunLog


class RunRepository:
    """Writes one JSON line per event to runs/<run_id>.jsonl.

    Thread/async-safety: best-effort. We rely on run_id uniqueness + append-only
    semantics. For multi-writer production, switch to a queue.
    """

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        # Defensive: avoid path traversal
        safe = "".join(c for c in run_id if c.isalnum() or c in "-_")
        if not safe or safe != run_id:
            raise ValueError(f"invalid run_id: {run_id!r}")
        return self.runs_dir / f"{safe}.jsonl"

    async def write_event(self, run_id: str, event_type: str, data: dict[str, Any]) -> None:
        path = self._path(run_id)
        # File I/O offloaded to a thread to avoid blocking the event loop.
        import asyncio

        def _write() -> None:
            with path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"ts": data.get("ts", 0.0), "type": event_type, "data": data},
                        ensure_ascii=False,
                    )
                )
                f.write("\n")

        await asyncio.to_thread(_write)

    async def read(self, run_id: str) -> RunLog | None:
        path = self._path(run_id)
        if not path.exists():
            return None
        import asyncio

        def _read() -> list[dict[str, Any]]:
            with path.open("r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]

        lines = await asyncio.to_thread(_read)
        if not lines:
            return RunLog(run_id=run_id, query="")

        events: list[RunEvent] = []
        query = ""
        llm_calls = tokens_in = tokens_out = 0
        latency_ms = 0
        errors: list[dict[str, Any]] = []
        final_count = 0
        for ln in lines:
            et = ln.get("type", "")
            d = ln.get("data", {}) or {}
            events.append(RunEvent(ts=ln.get("ts", 0.0), type=et, data=d))  # type: ignore[arg-type]
            if et == "run_start":
                query = d.get("query", "")
            if et == "parse_mandate":
                llm_calls = max(llm_calls, 1)
                tokens_in += int(d.get("tokens_in", 0))
                tokens_out += int(d.get("tokens_out", 0))
            if et == "validation":
                llm_calls = max(llm_calls, 1)  # best-effort
            if et == "run_end":
                final_count = int(d.get("final_count", 0))
                latency_ms = int(d.get("latency_ms", 0))
        return RunLog(
            run_id=run_id,
            query=query,
            events=events,
            final_count=final_count,
            llm_calls=llm_calls,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            errors=errors,
        )


__all__ = ["RunRepository"]
