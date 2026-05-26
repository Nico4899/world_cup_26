"""travel_km_diff column on match_features (W2.2)

Revision ID: c1d4f0e2a955
Revises: b8a2f9d04c11
Create Date: 2026-05-26 21:45:00.000000

Adds one nullable Float column to ``features_match_features``:

* ``travel_km_diff`` — ``home_travel_km - away_travel_km`` great-circle
  distance from each team's previous match venue to the current one
  (see :mod:`wc2026.features.travel`).

Nullable: feature naturally degrades to NaN for pre-tournament matches
where the venue coordinates aren't carried on the row. XGB's hist
tree-method handles NaN as a missing-value branch.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c1d4f0e2a955"
down_revision: str | Sequence[str] | None = "b8a2f9d04c11"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "features_match_features",
        sa.Column("travel_km_diff", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("features_match_features", "travel_km_diff")
