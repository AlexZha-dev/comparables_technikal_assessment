"""Bootstrap of runtime services (repositories, LLM client, workflow).

Kept in one place so:
- Tests can construct `RuntimeServices` directly without going through the
  FastAPI lifespan (no app fixture needed for unit tests).
- main.create_app() doesn't import a dozen concrete classes — it imports
  one function.
- The exact list of subsystems and their dependencies is visible at a glance.

Each subsystem is constructed in `bootstrap_subsystems(settings)`. Failures
are caught and reflected in the per-subsystem ready flags — `/health/ready`
uses those flags to report partial degradation.
"""
from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass

from comparables.core.config import Settings
from comparables.core.logging import get_logger
from comparables.db.session import Database
from comparables.llm.client import LLMClient
from comparables.repositories.bm25_repo import BM25Repository
from comparables.repositories.company_repo import CompanyRepository
from comparables.repositories.run_repo import RunRepository
from comparables.services.workflow_service import WorkflowService
from comparables.tools.registry import ToolRegistry, default_registry

logger = get_logger(__name__)


# ─── Aggregate of all running subsystems ──────────────────────────────
@dataclass
class RuntimeServices:
    """Everything we need to serve a request, in one immutable bundle.

    The constructor is intentionally minimal: it builds the objects but
    doesn't connect to anything. Call `await services.startup()` to actually
    initialize async resources (SQLAlchemy engine/pool, BM25 pickle).
    """

    settings: Settings
    tool_registry: ToolRegistry
    database: Database
    company_repo: CompanyRepository
    bm25_repo: BM25Repository
    run_repo: RunRepository
    llm_client: LLMClient

    # Computed at startup().
    workflow_service: WorkflowService | None = None
    sqlite_ready: bool = False
    bm25_loaded: bool = False
    llm_ready: bool = False

    async def startup(self) -> None:
        """Connect async resources (DB engine, BM25) and verify LLM reachable.

        Errors per subsystem are caught and reflected in the `*_ready` flags
        so the request-handling path can degrade gracefully — failed BM25 still
        serves a structured response that explains what's missing.
        """
        try:
            await self.company_repo.connect()
            self.sqlite_ready = True
            logger.info("startup.sqlite_ok", path=str(self.settings.paths.sqlite))
        except Exception as exc:
            self.sqlite_ready = False
            logger.warning("startup.sqlite_failed", error=str(exc))

        try:
            await self.bm25_repo.load()
            self.bm25_loaded = True
            logger.info("startup.bm25_ok", path=str(self.settings.paths.bm25_pickle))
        except Exception as exc:
            self.bm25_loaded = False
            logger.warning("startup.bm25_failed", error=str(exc))

        try:
            self.llm_ready = bool(await self.llm_client.ping())
            if self.llm_ready:
                logger.info("startup.llm_ok", model=self.settings.llm.model)
            else:
                logger.warning(
                    "startup.llm_unreachable", base_url=self.settings.llm.base_url
                )
        except Exception as exc:
            self.llm_ready = False
            logger.warning("startup.llm_failed", error=str(exc))

        # Workflow service doesn't do I/O until the first request.
        self.workflow_service = WorkflowService(
            settings=self.settings,
            company_repo=self.company_repo,
            bm25_repo=self.bm25_repo,
            llm=self.llm_client,
            run_repo=self.run_repo,
        )

    async def shutdown(self) -> None:
        """Close async resources. Errors are logged but never propagated."""
        with suppress(Exception):
            await self.database.shutdown()
        with suppress(Exception):
            await self.llm_client.aclose()


# ─── Factory ──────────────────────────────────────────────────────────
def bootstrap_subsystems(settings: Settings) -> RuntimeServices:
    """Construct all subsystems from settings. Doesn't touch I/O."""
    database = Database.from_settings(settings)
    return RuntimeServices(
        settings=settings,
        tool_registry=default_registry(),
        database=database,
        company_repo=CompanyRepository(db=database),
        bm25_repo=BM25Repository(path=settings.paths.bm25_pickle),
        run_repo=RunRepository(runs_dir=settings.paths.runs_dir),
        llm_client=LLMClient(
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
            model=settings.llm.model,
            timeout_s=settings.llm.timeout_s,
        ),
    )
