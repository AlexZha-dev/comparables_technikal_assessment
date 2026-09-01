"""Pydantic DTOs for the company catalog."""
from __future__ import annotations

from pydantic import BaseModel, Field


class CompanyRecord(BaseModel):
    """One row in the company catalog."""

    id: int
    name: str
    description: str
    industry: str
    location: str
    founded_year: int
    employee_count: int = Field(ge=0)
    revenue_range: str
