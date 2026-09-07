# Comparables.ai - app image
#
# Multi-stage build keeps the final image small while still caching deps.
# Base: python:3.11-slim (matches >=3.11 requirement; conservative pin).
#
# The supported operator path is `docker compose` (see README). Compose
# supplies the model, ingestion init job, persistent named volumes and the
# network name for Ollama. Keeping that wiring out of this image avoids
# hard-coded host mounts or `host.docker.internal` assumptions.

# ─── Stage 1: deps + build cache ──────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml README.md ./
# If you switch to a src layout with __init__.py only, pip install . works
# without an sdist step.
COPY src ./src
RUN pip install .

# Test dependencies are deliberately isolated from the runtime image. The
# `test` target below makes every documented check reproducible in Docker.
FROM builder AS test-deps
RUN pip install ".[dev]"

# ─── Stage 2: runtime ────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API__HOST=0.0.0.0 \
    API__PORT=8000 \
    API__LOG_LEVEL=INFO

# Non-root user
RUN groupadd --system app && useradd --system --gid app --home /app app
WORKDIR /app
RUN mkdir -p /app/data /app/runs && chown -R app:app /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src ./src
COPY scripts ./scripts
COPY --chown=app:app eval ./eval
COPY migrations ./migrations
COPY alembic.ini ./alembic.ini
COPY pyproject.toml README.md ./

# Entry point: alembic upgrade head, then uvicorn.
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER app

EXPOSE 8000

# A healthy container must have a usable catalogue/index, not merely a live
# Python process. LLM availability remains a soft signal in `/ready`.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health/ready', timeout=3).read()" \
    || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Test target inherits the non-root runtime contract and entrypoint. An
# explicit command is executed directly by entrypoint, without API startup.
FROM runtime AS test
COPY --from=test-deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=test-deps /usr/local/bin /usr/local/bin
COPY --chown=app:app tests ./tests
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]

# Keep `docker build .` production-safe: an unqualified build selects this
# runtime stage rather than the test image with development tooling.
FROM runtime AS production
