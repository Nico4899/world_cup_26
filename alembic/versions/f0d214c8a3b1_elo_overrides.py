"""elo_overrides

Revision ID: f0d214c8a3b1
Revises: e9b3a4f17d20
Create Date: 2026-05-24 16:00:00.000000

Adds ``raw_elo_overrides`` so operators can patch a single team's Elo
rating when the eloratings.net scraper is broken or returns obviously-wrong
values. Written + read by ``POST/GET/DELETE /api/v1/_ops/elo-override``.
The override is merged on top of the disk-side parquet snapshot at read
time (see ``wc2026.ingest.eloratings_scraper.load_latest_with_overrides``).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f0d214c8a3b1"
down_revision: str | Sequence[str] | None = "e9b3a4f17d20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "raw_elo_overrides",
        sa.Column("team_code", sa.String(length=8), primary_key=True),
        sa.Column("team_name", sa.String(length=128), nullable=True),
        sa.Column("rating", sa.Float(), nullable=False),
        sa.Column("reason", sa.String(length=256), nullable=True),
        sa.Column("set_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("raw_elo_overrides")
