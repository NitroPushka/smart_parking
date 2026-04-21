"""Add analysis tasks table

Revision ID: 2d4c1a7b8c9d
Revises: 1dfb2aec130a
Create Date: 2026-04-21 18:45:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "2d4c1a7b8c9d"
down_revision: Union[str, None] = "1dfb2aec130a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "analysis_tasks",
        sa.Column("task_id", sa.String(), nullable=False),
        sa.Column("lot_id", sa.Integer(), nullable=False),
        sa.Column("image_path", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=True),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=True),
        sa.ForeignKeyConstraint(["lot_id"], ["parking_lots.id"]),
        sa.PrimaryKeyConstraint("task_id"),
    )
    op.create_index(op.f("ix_analysis_tasks_task_id"), "analysis_tasks", ["task_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_analysis_tasks_task_id"), table_name="analysis_tasks")
    op.drop_table("analysis_tasks")
