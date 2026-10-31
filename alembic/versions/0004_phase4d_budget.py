"""add run_usage and budget_decisions

Revision ID: 0004_phase4d_budget
Revises: 0003_phase4b_uploads
Create Date: 2026-10-31
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0004_phase4d_budget"
down_revision: str | None = "0003_phase4b_uploads"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "run_usage",
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("spent_usd", sa.Numeric(12, 6), nullable=False, server_default="0"),
        sa.Column("reserved_usd", sa.Numeric(12, 6), nullable=False, server_default="0"),
        sa.Column("spent_wall_clock_s", sa.Float(), nullable=False, server_default="0"),
        sa.Column("target_requests", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("aig_requests", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_table(
        "budget_decisions",
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(length=32), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id", "seq"),
    )


def downgrade() -> None:
    op.drop_table("budget_decisions")
    op.drop_table("run_usage")
