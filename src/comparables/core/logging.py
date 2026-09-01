"""structlog setup: JSON in prod, pretty in dev; binds run_id from contextvar."""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from comparables.core.context import run_id_var
from comparables.core.config import Settings


def _add_run_id(_, __, event_dict: dict[str, Any]) -> dict[str, Any]:
    rid = run_id_var.get()
    if rid is not None:
        event_dict.setdefault("run_id", rid)
    return event_dict


def configure_logging(settings: Settings) -> None:
    """Configure stdlib logging + structlog once at startup.

    Idempotent: safe to call multiple times.
    """
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _add_run_id,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]

    if settings.log_json and not settings.is_dev:
        processors.extend(
            [
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ]
        )
    else:
        processors.extend(
            [
                structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
            ]
        )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
