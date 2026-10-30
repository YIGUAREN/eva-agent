"""Web foundation: run metadata, commands, sessions, reports, artifacts."""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_phase4a_web"
down_revision: Union[str, None] = "0001_baseline"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_ACTIVE = "status IN ('pending','claimed') AND kind != 'cancel'"


def upgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        batch.add_column(sa.Column("idempotency_key", sa.String(128), nullable=True))
        batch.add_column(sa.Column("authorization_ref", sa.String(256), nullable=True))
        batch.add_column(sa.Column("error", sa.Text(), nullable=True))
        batch.add_column(sa.Column("created_by", sa.String(32), nullable=True))
        batch.add_column(sa.Column("published_at", sa.DateTime(timezone=True), nullable=True))
        batch.create_unique_constraint("uq_runs_idempotency_key", ["idempotency_key"])

    op.create_table(
        "run_commands",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_run_commands_run_id", "run_commands", ["run_id"])
    op.create_index("ix_run_commands_status_id", "run_commands", ["status", "id"])
    op.create_index(
        "uq_run_commands_active",
        "run_commands",
        ["run_id"],
        unique=True,
        sqlite_where=sa.text(_ACTIVE),
        postgresql_where=sa.text(_ACTIVE),
    )

    op.create_table(
        "admin_sessions",
        sa.Column("id_hash", sa.String(64), primary_key=True),
        sa.Column("csrf_token", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("absolute_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_admin_sessions_expires", "admin_sessions", ["expires_at"])

    op.create_table(
        "reports",
        sa.Column("run_id", sa.String(36), sa.ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("json", sa.JSON(), nullable=False),
        sa.Column("markdown", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "artifacts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("content_type", sa.String(128), nullable=False),
        sa.Column("visibility", sa.String(8), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("run_id", "name", name="uq_artifacts_run_name"),
    )


def downgrade() -> None:
    op.drop_table("artifacts")
    op.drop_table("reports")
    op.drop_index("ix_admin_sessions_expires", table_name="admin_sessions")
    op.drop_table("admin_sessions")
    op.drop_index("uq_run_commands_active", table_name="run_commands")
    op.drop_index("ix_run_commands_status_id", table_name="run_commands")
    op.drop_index("ix_run_commands_run_id", table_name="run_commands")
    op.drop_table("run_commands")
    with op.batch_alter_table("runs") as batch:
        batch.drop_constraint("uq_runs_idempotency_key", type_="unique")
        batch.drop_column("published_at")
        batch.drop_column("created_by")
        batch.drop_column("error")
        batch.drop_column("authorization_ref")
        batch.drop_column("idempotency_key")
