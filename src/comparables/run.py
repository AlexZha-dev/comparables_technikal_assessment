"""ASGI entry point: `python -m comparables.run` (or `uvicorn comparables.run:run`).

This is the production boot file: it reads settings, builds the app via the
factory, and hands it to uvicorn. Keeping this separate from `main.py` means
domain code can be imported without ever touching uvicorn — useful in tests.

CLI usage:
    python -m comparables.run                 # default 0.0.0.0:8000
    python -m comparables.run --port 9000     # custom port
    python -m comparables.run --reload        # dev autoreload

Programmatic usage (e.g. wsgi server in front of nginx):
    from comparables.run import run
    uvicorn.run(run, host="0.0.0.0", port=8000)
"""
from __future__ import annotations

import argparse

import uvicorn

from comparables.core.config import get_settings
from comparables.main import create_app

# A module-level `app` is what uvicorn expects when given a string import.
app = create_app()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Comparables.ai API server")
    p.add_argument("--host", default=None, help="bind host (default from settings)")
    p.add_argument("--port", type=int, default=None, help="bind port (default from settings)")
    p.add_argument("--reload", action="store_true", help="enable dev autoreload")
    p.add_argument("--log-level", default=None, help="uvicorn log level")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    args = _parse_args(argv)
    uvicorn.run(
        "comparables.run:app",
        host=args.host or settings.app_host,
        port=args.port or settings.app_port,
        reload=args.reload,
        log_level=args.log_level or settings.log_level.lower(),
        factory=False,
    )


if __name__ == "__main__":
    main()
