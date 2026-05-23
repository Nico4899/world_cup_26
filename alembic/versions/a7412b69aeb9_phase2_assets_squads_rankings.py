"""phase2_assets_squads_rankings

Revision ID: a7412b69aeb9
Revises: 07715e364a9c
Create Date: 2026-05-23 22:30:00.000000

Adds three tables that back the Phase 2 ingesters:

* ``raw_team_assets``    — TheSportsDB crest / kit / stadium metadata
* ``raw_squads``         — Wikipedia tournament-squad rosters (history-preserving)
* ``raw_fifa_rankings``  — Wikipedia FIFA Men's Ranking snapshots
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a7412b69aeb9"
down_revision: str | Sequence[str] | None = "07715e364a9c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "raw_team_assets",
        sa.Column("team", sa.String(length=128), nullable=False),
        sa.Column("thesportsdb_id", sa.Integer(), nullable=True),
        sa.Column("crest_url", sa.String(length=512), nullable=True),
        sa.Column("kit_home_color", sa.String(length=16), nullable=True),
        sa.Column("kit_away_color", sa.String(length=16), nullable=True),
        sa.Column("stadium_name", sa.String(length=128), nullable=True),
        sa.Column("stadium_capacity", sa.Integer(), nullable=True),
        sa.Column("stadium_city", sa.String(length=128), nullable=True),
        sa.Column("stadium_country", sa.String(length=128), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("team"),
    )

    op.create_table(
        "raw_squads",
        sa.Column("tournament", sa.String(length=128), nullable=False),
        sa.Column("team", sa.String(length=128), nullable=False),
        sa.Column("player_name", sa.String(length=128), nullable=False),
        sa.Column("snapshot_date", sa.Date(), nullable=False),
        sa.Column("shirt_number", sa.Integer(), nullable=True),
        sa.Column("position", sa.String(length=8), nullable=True),
        sa.Column("birth_date", sa.Date(), nullable=True),
        sa.Column("club", sa.String(length=128), nullable=True),
        sa.Column("caps", sa.Integer(), nullable=True),
        sa.Column("goals", sa.Integer(), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("tournament", "team", "player_name", "snapshot_date"),
    )
    op.create_index("ix_raw_squads_team", "raw_squads", ["team"])
    op.create_index("ix_raw_squads_snapshot_date", "raw_squads", ["snapshot_date"])

    op.create_table(
        "raw_fifa_rankings",
        sa.Column("ranking_date", sa.Date(), nullable=False),
        sa.Column("team", sa.String(length=128), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("points", sa.Float(), nullable=True),
        sa.Column("previous_rank", sa.Integer(), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("ranking_date", "team"),
    )
    op.create_index("ix_raw_fifa_rankings_team", "raw_fifa_rankings", ["team"])


def downgrade() -> None:
    op.drop_index("ix_raw_fifa_rankings_team", table_name="raw_fifa_rankings")
    op.drop_table("raw_fifa_rankings")
    op.drop_index("ix_raw_squads_snapshot_date", table_name="raw_squads")
    op.drop_index("ix_raw_squads_team", table_name="raw_squads")
    op.drop_table("raw_squads")
    op.drop_table("raw_team_assets")
