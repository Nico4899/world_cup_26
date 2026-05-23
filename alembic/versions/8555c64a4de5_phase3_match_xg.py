"""phase3_match_xg

Revision ID: 8555c64a4de5
Revises: a7412b69aeb9
Create Date: 2026-05-23 22:50:00.000000

Adds the ``raw_match_xg`` per-match-per-team expected-goals aggregate. Per-shot
events stay in Parquet snapshots under ``data/raw/statsbomb/`` — Postgres only
holds the summary used by feature engineering.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "8555c64a4de5"
down_revision: str | Sequence[str] | None = "a7412b69aeb9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "raw_match_xg",
        sa.Column("match_date", sa.Date(), nullable=False),
        sa.Column("home_team", sa.String(length=128), nullable=False),
        sa.Column("away_team", sa.String(length=128), nullable=False),
        sa.Column("team", sa.String(length=128), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("xg_for", sa.Float(), nullable=False),
        sa.Column("xg_against", sa.Float(), nullable=False),
        sa.Column("shots", sa.Integer(), nullable=True),
        sa.Column("shots_on_target", sa.Integer(), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("match_date", "home_team", "away_team", "team", "source"),
    )
    op.create_index("ix_raw_match_xg_team", "raw_match_xg", ["team"])
    op.create_index("ix_raw_match_xg_match_date", "raw_match_xg", ["match_date"])


def downgrade() -> None:
    op.drop_index("ix_raw_match_xg_match_date", table_name="raw_match_xg")
    op.drop_index("ix_raw_match_xg_team", table_name="raw_match_xg")
    op.drop_table("raw_match_xg")
