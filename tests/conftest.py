"""Shared pytest fixtures."""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from comparables.core.config import get_settings, reset_settings_cache
from comparables.ingestion.index import run_ingestion
from comparables.main import create_app


@pytest.fixture(scope="session", autouse=True)
def isolated_catalog(tmp_path_factory):
    """Build a tiny catalog through the real ingestion path, never touch user data."""
    directory = tmp_path_factory.mktemp("catalog")
    records = [
        {"id": 1, "name": "Nordic Fintech Solutions", "description": "AI-powered platform for fraud detection, banking analytics, and risk assessment.", "industry": "Fintech", "location": "Finland", "employee_count": 250, "founded_year": 2020, "revenue_range": "10M-50M"},
        {"id": 2, "name": "Baltic Payments Cloud", "description": "Cloud-native payments infrastructure and financial data platform for digital banking.", "industry": "Fintech", "location": "Finland", "employee_count": 50, "founded_year": 2015, "revenue_range": "1M-10M"},
        {"id": 3, "name": "Solar Startup", "description": "Renewable energy startup for solar installations.", "industry": "Energy", "location": "Germany", "employee_count": 150, "founded_year": 2020, "revenue_range": "10M-50M"},
        {"id": 4, "name": "Mobility Lab", "description": "Autonomous driving using machine learning.", "industry": "Automotive", "location": "Netherlands", "employee_count": 200, "founded_year": 2019, "revenue_range": "10M-50M"},
        {"id": 5, "name": "Care Group", "description": "Healthcare services and clinical diagnostics.", "industry": "Healthcare", "location": "USA", "employee_count": 1000, "founded_year": 2010, "revenue_range": "50M-100M"},
    ]
    source = directory / "companies.json"
    source.write_text(json.dumps(records), encoding="utf-8")
    with pytest.MonkeyPatch.context() as patch:
        for key, path in {
            "PATHS__COMPANIES_JSON": source,
            "PATHS__SQLITE": directory / "companies.sqlite",
            "PATHS__BM25_PICKLE": directory / "bm25.pkl",
            "PATHS__RUNS_DIR": directory / "runs",
        }.items():
            patch.setenv(key, str(path))
        reset_settings_cache()
        asyncio.run(run_ingestion())
        yield get_settings()
        reset_settings_cache()


@pytest_asyncio.fixture
async def app():
    application = create_app()
    yield application
    # No explicit close — lifespan handles it via ASGITransport


@pytest_asyncio.fixture
async def client(app) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
