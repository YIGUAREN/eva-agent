"""add uploads table

Revision ID: 0003_phase4b_uploads
Revises: 0002_phase4a_web
Create Date: 2026-10-31
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003_phase4b_uploads"
down_revision: str | None = "0002_phase4a_web"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "uploads",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("format", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("manifest_json", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_uploads_status", "uploads", ["status"])
    op.create_index("ix_uploads_expires_at", "uploads", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_uploads_expires_at", table_name="uploads")
    op.drop_index("ix_uploads_status", table_name="uploads")
    op.drop_table("uploads")
