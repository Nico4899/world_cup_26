"""Load raw Jürisoo results.csv and the latest Elo snapshot parquet into Postgres.

Usage:
    uv run python scripts/db_load_from_parquet.py [--results-dir PATH] [--elo-dir PATH]

Skips rows whose (date, home_team, away_team, source) natural key is already loaded
for the same source, so the script is safe to re-run nightly.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from wc2026.db.models import RawEloSnapshot, RawMatch
from wc2026.db.session import session_scope
from wc2026.ingest.eloratings_scraper import DEFAULT_TARGET as ELO_DEFAULT
from wc2026.ingest.kaggle_intl import DEFAULT_TARGET as JURISOO_DEFAULT
from wc2026.ingest.kaggle_intl import load_results

JURISOO_SOURCE = "jurisoo_kaggle"


def _latest_elo_parquet(elo_dir: Path) -> Path:
    paths = sorted(elo_dir.glob("elo_current_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No elo_current_*.parquet found in {elo_dir}")
    return paths[-1]


def _upsert_matches(df: pd.DataFrame, *, source: str = JURISOO_SOURCE) -> int:
    """Insert raw_matches rows with ON CONFLICT DO NOTHING on the natural key.

    Returns the number of rows attempted (not necessarily inserted, since duplicates
    are quietly ignored). Postgres-specific; for SQLite fall back to the slower
    SELECT-then-INSERT path via _upsert_matches_generic.
    """
    payload = []
    now = datetime.now(UTC)
    for row in df.itertuples(index=False):
        payload.append(
            {
                "date": row.date.date() if hasattr(row.date, "date") else row.date,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "home_score": None if pd.isna(row.home_score) else int(row.home_score),
                "away_score": None if pd.isna(row.away_score) else int(row.away_score),
                "tournament": row.tournament,
                "city": None if pd.isna(row.city) else row.city,
                "country": None if pd.isna(row.country) else row.country,
                "neutral": bool(row.neutral),
                "source": source,
                "ingested_at": now,
            }
        )
    if not payload:
        return 0
    with session_scope() as s:
        stmt = pg_insert(RawMatch).values(payload)
        stmt = stmt.on_conflict_do_nothing(constraint="uq_raw_matches_natural_key")
        s.execute(stmt)
    return len(payload)


def _upsert_elo_snapshot(df: pd.DataFrame) -> int:
    payload = []
    for row in df.itertuples(index=False):
        snap = row.snapshot_date
        if hasattr(snap, "date"):
            snap = snap.date()
        payload.append(
            {
                "snapshot_date": snap,
                "team_code": row.code,
                "team_name": None if pd.isna(getattr(row, "team_name", None)) else row.team_name,
                "rating": float(row.rating),
                "global_rank": None if pd.isna(row.global_rank) else int(row.global_rank),
            }
        )
    if not payload:
        return 0
    with session_scope() as s:
        stmt = pg_insert(RawEloSnapshot).values(payload)
        stmt = stmt.on_conflict_do_update(
            index_elements=["snapshot_date", "team_code"],
            set_={
                "team_name": stmt.excluded.team_name,
                "rating": stmt.excluded.rating,
                "global_rank": stmt.excluded.global_rank,
            },
        )
        s.execute(stmt)
    return len(payload)


def load_all(results_dir: Path = JURISOO_DEFAULT, elo_dir: Path = ELO_DEFAULT) -> dict[str, int]:
    """Top-level orchestration: load Jürisoo results + latest Elo snapshot."""
    matches_df = load_results(results_dir)
    n_matches = _upsert_matches(matches_df)

    elo_path = _latest_elo_parquet(elo_dir)
    elo_df = pd.read_parquet(elo_path)
    n_elo = _upsert_elo_snapshot(elo_df)

    return {"matches_attempted": n_matches, "elo_rows_attempted": n_elo}


def _verify_counts() -> dict[str, int]:
    with session_scope() as s:
        n_matches = s.scalar(select(RawMatch.id).order_by(RawMatch.id.desc()).limit(1)) or 0
        n_elo = s.scalar(select(RawEloSnapshot.team_code).limit(1))
    return {"max_match_id": int(n_matches), "any_elo_row": str(n_elo) if n_elo else ""}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=JURISOO_DEFAULT)
    parser.add_argument("--elo-dir", type=Path, default=ELO_DEFAULT)
    args = parser.parse_args()

    counts = load_all(results_dir=args.results_dir, elo_dir=args.elo_dir)
    print(f"Load complete: {counts}")
    print(f"Verification: {_verify_counts()}")


if __name__ == "__main__":
    main()
