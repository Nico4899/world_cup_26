"""phase4_match_features

Revision ID: d513c23fed82
Revises: 8555c64a4de5
Create Date: 2026-05-23 23:10:00.000000

Adds the materialised ``features_match_features`` table. One row per
(match_date, home_team, away_team), holding every numeric input Phase 5's
XGBoost classifier consumes plus a JSON ``source_snapshots`` provenance blob.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d513c23fed82"
down_revision: str | Sequence[str] | None = "8555c64a4de5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "features_match_features",
        sa.Column("match_date", sa.Date(), nullable=False),
        sa.Column("home_team", sa.String(length=128), nullable=False),
        sa.Column("away_team", sa.String(length=128), nullable=False),
        sa.Column("elo_diff", sa.Float(), nullable=True),
        sa.Column("fifa_rank_diff", sa.Float(), nullable=True),
        sa.Column("xg_form_diff", sa.Float(), nullable=True),
        sa.Column("rest_days_diff", sa.Float(), nullable=True),
        sa.Column("squad_age_diff", sa.Float(), nullable=True),
        sa.Column("is_neutral", sa.Integer(), nullable=True),
        sa.Column("is_host_home", sa.Integer(), nullable=True),
        sa.Column("is_host_away", sa.Integer(), nullable=True),
        sa.Column("poisson_exp_home_goals", sa.Float(), nullable=True),
        sa.Column("poisson_exp_away_goals", sa.Float(), nullable=True),
        sa.Column("poisson_p_home", sa.Float(), nullable=True),
        sa.Column("poisson_p_draw", sa.Float(), nullable=True),
        sa.Column("poisson_p_away", sa.Float(), nullable=True),
        sa.Column("source_snapshots", sa.JSON(), nullable=True),
        sa.Column("built_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("match_date", "home_team", "away_team"),
    )
    op.create_index(
        "ix_features_match_features_match_date",
        "features_match_features",
        ["match_date"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_features_match_features_match_date",
        table_name="features_match_features",
    )
    op.drop_table("features_match_features")
