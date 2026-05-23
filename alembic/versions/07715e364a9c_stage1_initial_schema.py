"""stage1_initial_schema

Revision ID: 07715e364a9c
Revises:
Create Date: 2026-05-23 18:18:32.346882

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "07715e364a9c"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "raw_matches",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("home_team", sa.String(length=128), nullable=False),
        sa.Column("away_team", sa.String(length=128), nullable=False),
        sa.Column("home_score", sa.Integer(), nullable=True),
        sa.Column("away_score", sa.Integer(), nullable=True),
        sa.Column("tournament", sa.String(length=128), nullable=False),
        sa.Column("city", sa.String(length=128), nullable=True),
        sa.Column("country", sa.String(length=128), nullable=True),
        sa.Column("neutral", sa.Boolean(), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "date", "home_team", "away_team", "source", name="uq_raw_matches_natural_key"
        ),
    )
    op.create_index("ix_raw_matches_date", "raw_matches", ["date"])
    op.create_index("ix_raw_matches_home_team", "raw_matches", ["home_team"])
    op.create_index("ix_raw_matches_away_team", "raw_matches", ["away_team"])
    op.create_index("ix_raw_matches_source", "raw_matches", ["source"])

    op.create_table(
        "raw_elo_snapshots",
        sa.Column("snapshot_date", sa.Date(), nullable=False),
        sa.Column("team_code", sa.String(length=8), nullable=False),
        sa.Column("team_name", sa.String(length=128), nullable=True),
        sa.Column("rating", sa.Float(), nullable=False),
        sa.Column("global_rank", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("snapshot_date", "team_code"),
    )

    op.create_table(
        "model_predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("match_date", sa.Date(), nullable=False),
        sa.Column("home_team", sa.String(length=128), nullable=False),
        sa.Column("away_team", sa.String(length=128), nullable=False),
        sa.Column("p_home", sa.Float(), nullable=False),
        sa.Column("p_draw", sa.Float(), nullable=False),
        sa.Column("p_away", sa.Float(), nullable=False),
        sa.Column("score_matrix_json", sa.JSON(), nullable=True),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_predictions_match_date", "model_predictions", ["match_date"])
    op.create_index("ix_model_predictions_model_version", "model_predictions", ["model_version"])

    op.create_table(
        "tournament_sim_runs",
        sa.Column("run_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("n_sims", sa.Integer(), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("run_id"),
    )

    op.create_table(
        "tournament_sim_team_outcomes",
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("team", sa.String(length=128), nullable=False),
        sa.Column("group_winner_p", sa.Float(), nullable=False),
        sa.Column("group_runner_up_p", sa.Float(), nullable=False),
        sa.Column("advance_r32_p", sa.Float(), nullable=False),
        sa.Column("advance_r16_p", sa.Float(), nullable=False),
        sa.Column("quarterfinal_p", sa.Float(), nullable=False),
        sa.Column("semifinal_p", sa.Float(), nullable=False),
        sa.Column("final_p", sa.Float(), nullable=False),
        sa.Column("champion_p", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["tournament_sim_runs.run_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("run_id", "team"),
    )

    op.create_table(
        "scheduler_job_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_name", sa.String(length=128), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("error_text", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_scheduler_job_runs_job_name", "scheduler_job_runs", ["job_name"])
    op.create_index("ix_scheduler_job_runs_status", "scheduler_job_runs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_scheduler_job_runs_status", table_name="scheduler_job_runs")
    op.drop_index("ix_scheduler_job_runs_job_name", table_name="scheduler_job_runs")
    op.drop_table("scheduler_job_runs")
    op.drop_table("tournament_sim_team_outcomes")
    op.drop_table("tournament_sim_runs")
    op.drop_index("ix_model_predictions_model_version", table_name="model_predictions")
    op.drop_index("ix_model_predictions_match_date", table_name="model_predictions")
    op.drop_table("model_predictions")
    op.drop_table("raw_elo_snapshots")
    op.drop_index("ix_raw_matches_source", table_name="raw_matches")
    op.drop_index("ix_raw_matches_away_team", table_name="raw_matches")
    op.drop_index("ix_raw_matches_home_team", table_name="raw_matches")
    op.drop_index("ix_raw_matches_date", table_name="raw_matches")
    op.drop_table("raw_matches")
