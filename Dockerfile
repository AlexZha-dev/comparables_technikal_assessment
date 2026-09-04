# Comparables.ai - app image
#
# Two-stage build keeps the final image small while still caching deps.
# Base: python:3.11-slim (matches >=3.11 requirement; conservative pin).
#
# Build:   docker build -t comparables-app .
# Run:     docker run --rm -p 8000:8000 \
#             -v ${PWD}/data:/app/data \
#             -v ${PWD}/runs:/app/runs \
#             -v ${PWD}/companies.json:/app/companies.json:ro \
#             -e OLLAMA_BASE_URL=http://host.docker.internal:11434/v1 \
#             comparables-app

# ─── Stage 1: deps + build cache ──────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install build deps for any wheels that need compiling (rank-bm25, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
# If you switch to a src layout with __init__.py only, pip install . works
# without an sdist step.
COPY src ./src
RUN pip install --upgrade pip && pip install .

# ─── Stage 2: runtime ────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    LOG_LEVEL=INFO

# Non-root user
RUN groupadd --system app && useradd --system --gid app --home /app app
WORKDIR /app
RUN mkdir -p /app/data /app/runs && chown -R app:app /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src ./src
COPY scripts ./scripts
COPY pyproject.toml ./

USER app

EXPOSE 8000

# Healthcheck uses the liveness endpoint defined in src/comparables/api/v1/endpoints/health.py
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health/live', timeout=3).read()" \
    || exit 1

CMD ["uvicorn", "comparables.main:app", "--host", "0.0.0.0", "--port", "8000"]
