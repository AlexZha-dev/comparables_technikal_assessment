"""ASGI entry point.

Two ways to launch:

    1. `python -m comparables.run`            — module form
    2. `python src/comparables/run.py`        — direct script form

The direct form needs `src/` on `sys.path` so relative-ish imports work; we
patch that on first import. Module-form works because `comparableS` is already
on the path after `pip install -e .`.

All knobs (host, port, reload, log level) come from typed `Settings`. CLI
overrides are optional; precedence is `CLI > env > .env > defaults`.
"""
from __future__ import annotations

import argparse
import sys
from logging import getLogger
from pathlib import Path

# Make the project importable when running this file directly
# (`python src/comparables/run.py`). Module-form (`-m`) doesn't need this.
_THIS = Path(__file__).resolve()
_SRC = _THIS.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import uvicorn  # noqa: E402 — after sys.path tweak

from comparables.core.config import get_settings  # noqa: E402
from comparables.core.logging import configure_logging  # noqa: E402
from comparables.main import create_app  # noqa: E402

logger = getLogger("comparables.run")

# Module-level ASGI app — what uvicorn imports from the string path.
app = create_app()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Optional CLI overrides. Everything has a sensible default in Settings."""
    p = argparse.ArgumentParser(description="Comparables.ai API server")
    p.add_argument("--host", default=None, help="bind host (default: settings.api.host)")
    p.add_argument("--port", type=int, default=None, help="bind port (default: settings.api.port)")
    p.add_argument("--reload", action="store_true", default=None, help="enable dev autoreload (default: settings.api.reload)")
    p.add_argument(
        "--log-level", default=None, help="uvicorn log level (default: settings.api.log_level)"
    )
    # `store_true` flag defaulting to None lets us distinguish "user didn't pass it"
    # from "user passed --reload=False" (impossible via store_true, so we treat None
    # as 'fall through to settings').
    args = p.parse_args(argv)
    if args.reload is None:
        args.reload = False  # argparse quirk: action=store_true defaults to False;
                             # we want `None` semantics, so re-derive from settings.
    return args


def main(argv: list[str] | None = None) -> None:
    """Resolve overrides, start uvicorn."""
    settings = get_settings()
    configure_logging(settings)

    args = _parse_args(argv)
    host = args.host or settings.api.host
    port = args.port or settings.api.port
    reload_enabled = bool(settings.api.reload or args.reload)
    log_level = (args.log_level or settings.api.log_level).lower()

    logger.info(
        "uvicorn_start host=%s port=%s reload=%s log_level=%s",
        host,
        port,
        reload_enabled,
        log_level,
    )
    uvicorn.run(
        "comparables.run:app",
        host=host,
        port=port,
        reload=reload_enabled,
        log_level=log_level,
        factory=False,
    )


if __name__ == "__main__":
    main()
