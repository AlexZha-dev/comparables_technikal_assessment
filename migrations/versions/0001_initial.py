"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-01

Creates the `companies` table and the indexes the runtime query path
depends on (industry / location / revenue_range / employee_count /
founded_year). Mirrors `db/models/company.py` so SQLAlchemy autogenerate
sees a clean diff between the live DB and the model.

Uses `op.create_table` / `op.create_index` (the canonical Alembic DSL)
rather than raw `op.execute("CREATE TABLE …")` so autogenerate can
recognize and reverse these operations.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "companies",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("industry", sa.Text(), nullable=False),
        sa.Column("location", sa.Text(), nullable=False),
        sa.Column("founded_year", sa.Integer(), nullable=False),
        sa.Column("employee_count", sa.Integer(), nullable=False),
        sa.Column("revenue_range", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_company_industry", "companies", ["industry"])
    op.create_index("idx_company_location", "companies", ["location"])
    op.create_index("idx_company_revenue", "companies", ["revenue_range"])
    op.create_index("idx_company_employee_count", "companies", ["employee_count"])
    op.create_index("idx_company_founded_year", "companies", ["founded_year"])


def downgrade() -> None:
    # Drop indexes before the table to keep the rollback order explicit.
    op.drop_index("idx_company_founded_year", table_name="companies")
    op.drop_index("idx_company_employee_count", table_name="companies")
    op.drop_index("idx_company_revenue", table_name="companies")
    op.drop_index("idx_company_location", table_name="companies")
    op.drop_index("idx_company_industry", table_name="companies")
    op.drop_table("companies")
