"""phase12_team_outcome_third_out_fourth

Revision ID: e9b3a4f17d20
Revises: c312c32b3f8b
Create Date: 2026-05-24 12:00:00.000000

Adds ``third_out_p`` + ``fourth_p`` to ``tournament_sim_team_outcomes`` so the
5-segment group-stage bars (1st / 2nd / 3rd-advance / 3rd-out / 4th) can be
served from a persisted Monte Carlo run rather than re-derived in the
dashboard. Both columns are nullable so pre-Phase-12 rows still load; the
route degrades to a single "eliminated" bucket when the values are null.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e9b3a4f17d20"
down_revision: str | Sequence[str] | None = "c312c32b3f8b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "tournament_sim_team_outcomes",
        sa.Column("third_out_p", sa.Float(), nullable=True),
    )
    op.add_column(
        "tournament_sim_team_outcomes",
        sa.Column("fourth_p", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("tournament_sim_team_outcomes", "fourth_p")
    op.drop_column("tournament_sim_team_outcomes", "third_out_p")
