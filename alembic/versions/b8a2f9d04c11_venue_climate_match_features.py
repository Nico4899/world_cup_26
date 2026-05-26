"""venue climate columns on match_features (W2.1)

Revision ID: b8a2f9d04c11
Revises: a4c0e1f2b3d4
Create Date: 2026-05-26 21:00:00.000000

Adds two nullable Float columns to ``features_match_features``:

* ``venue_altitude_m``  — match-venue altitude in metres (e.g. Estadio Azteca = 2240).
* ``venue_wet_bulb_c``  — wet-bulb temperature forecast at kickoff time.

Both come from :mod:`wc2026.features.venue` (static lookup or Open-Meteo
forecast with climate-normal fallback). Existing rows stay NULL until the
next ``features_rebuild`` run; XGB's ``hist`` tree method treats NaN as a
natural missing-value branch, so the existing v1 artifact keeps working
even before a retrain.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b8a2f9d04c11"
down_revision: str | Sequence[str] | None = "a4c0e1f2b3d4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "features_match_features",
        sa.Column("venue_altitude_m", sa.Float(), nullable=True),
    )
    op.add_column(
        "features_match_features",
        sa.Column("venue_wet_bulb_c", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("features_match_features", "venue_wet_bulb_c")
    op.drop_column("features_match_features", "venue_altitude_m")
