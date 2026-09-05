#!/bin/sh
# Container entrypoint for the FastAPI service.
#
# Order matters:
#   1. alembic upgrade head   — bring the DB schema to the current revision.
#                               Fails fast on a broken migration; the
#                               container exits non-zero so an orchestrator
#                               (Compose / k8s) can restart it / roll back.
#   2. exec uvicorn           — replaces the shell so signals (SIGTERM from
#                               `docker stop`) reach the uvicorn process
#                               and it can drain in-flight requests.
#
# Explicit commands support one-shot ingestion, evaluation and diagnostics.
# With no command, migrate the database and start the API.

set -eu

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

echo "[entrypoint] alembic upgrade head"
python -m alembic upgrade head

echo "[entrypoint] starting FastAPI"
exec python -m comparables.run
