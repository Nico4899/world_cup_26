"""log_market_value_diff column on match_features (W2.3)

Revision ID: d2e5a6c83770
Revises: c1d4f0e2a955
Create Date: 2026-05-26 22:15:00.000000

Adds one nullable Float column to ``features_match_features``:

* ``log_market_value_diff`` — ``log(home_market_value_eur) - log(away_market_value_eur)``
  using the per-team squad totals scraped via :mod:`wc2026.ingest.transfermarkt`.

Nullable: feature is ``None`` until the manual ``transfermarkt_refresh``
job has produced at least one parquet snapshot, and ``None`` for any
team missing from that snapshot.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d2e5a6c83770"
down_revision: str | Sequence[str] | None = "c1d4f0e2a955"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "features_match_features",
        sa.Column("log_market_value_diff", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("features_match_features", "log_market_value_diff")
