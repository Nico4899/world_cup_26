"""phase6_live_events

Revision ID: c312c32b3f8b
Revises: d513c23fed82
Create Date: 2026-05-24 00:00:00.000000

Adds the ``raw_live_events`` table — one row per observed state-changing
event in a live football-data.org match (goal / red / yellow / sub / period
end). The schema captures the *post-event* score + red-card counts so the
Phase 6 live win-prob model can score directly off any row.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c312c32b3f8b"
down_revision: str | Sequence[str] | None = "d513c23fed82"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "raw_live_events",
        sa.Column("match_id", sa.Integer(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("minute", sa.Integer(), nullable=False),
        sa.Column("period", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("team", sa.String(length=128), nullable=True),
        sa.Column("player", sa.String(length=128), nullable=True),
        sa.Column("home_score_after", sa.Integer(), nullable=False),
        sa.Column("away_score_after", sa.Integer(), nullable=False),
        sa.Column("home_red_cards_after", sa.Integer(), nullable=False),
        sa.Column("away_red_cards_after", sa.Integer(), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("match_id", "seq"),
    )
    op.create_index(
        "ix_raw_live_events_event_type", "raw_live_events", ["event_type"]
    )


def downgrade() -> None:
    op.drop_index("ix_raw_live_events_event_type", table_name="raw_live_events")
    op.drop_table("raw_live_events")
