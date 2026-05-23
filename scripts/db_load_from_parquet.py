"""Load raw Jürisoo results.csv and the latest Elo snapshot parquet into Postgres.

Usage:
    uv run python scripts/db_load_from_parquet.py [--results-dir PATH] [--elo-dir PATH]

Skips rows whose (date, home_team, away_team, source) natural key is already loaded
for the same source, so the script is safe to re-run nightly.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from wc2026.db.models import RawEloSnapshot, RawMatch
from wc2026.db.session import session_scope
from wc2026.ingest.eloratings_scraper import DEFAULT_TARGET as ELO_DEFAULT
from wc2026.ingest.kaggle_intl import DEFAULT_TARGET as JURISOO_DEFAULT
from wc2026.ingest.kaggle_intl import load_results

JURISOO_SOURCE = "jurisoo_kaggle"

# Postgres caps a single statement at 65535 bound parameters. raw_matches has 11
# columns, raw_elo_snapshots has 5; 5000 rows * 11 = 55000 stays safely below the
# limit and lets the same constant cover both tables.
INSERT_BATCH_ROWS = 5000


def _latest_elo_parquet(elo_dir: Path) -> Path:
    paths = sorted(elo_dir.glob("elo_current_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No elo_current_*.parquet found in {elo_dir}")
    return paths[-1]


def _chunks(payload: list[dict], size: int) -> Iterable[list[dict]]:
    for i in range(0, len(payload), size):
        yield payload[i : i + size]


def _row_to_match_payload(row, *, source: str, now: datetime) -> dict:
    return {
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


def _row_to_elo_payload(row) -> dict:
    snap = row.snapshot_date
    if hasattr(snap, "date"):
        snap = snap.date()
    return {
        "snapshot_date": snap,
        "team_code": row.code,
        "team_name": None if pd.isna(getattr(row, "team_name", None)) else row.team_name,
        "rating": float(row.rating),
        "global_rank": None if pd.isna(row.global_rank) else int(row.global_rank),
    }


def _upsert_matches(
    df: pd.DataFrame, *, source: str = JURISOO_SOURCE, batch_size: int = INSERT_BATCH_ROWS
) -> int:
    """Insert raw_matches in batches with ON CONFLICT DO NOTHING on the natural key.

    Returns the number of rows attempted (not necessarily inserted, since duplicates
    are quietly ignored). Postgres-specific.
    """
    now = datetime.now(UTC)
    payload = [
        _row_to_match_payload(row, source=source, now=now) for row in df.itertuples(index=False)
    ]
    if not payload:
        return 0
    with session_scope() as s:
        for chunk in _chunks(payload, batch_size):
            stmt = pg_insert(RawMatch).values(chunk)
            stmt = stmt.on_conflict_do_nothing(constraint="uq_raw_matches_natural_key")
            s.execute(stmt)
    return len(payload)


def _upsert_elo_snapshot(df: pd.DataFrame, *, batch_size: int = INSERT_BATCH_ROWS) -> int:
    payload = [_row_to_elo_payload(row) for row in df.itertuples(index=False)]
    if not payload:
        return 0
    with session_scope() as s:
        for chunk in _chunks(payload, batch_size):
            stmt = pg_insert(RawEloSnapshot).values(chunk)
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
        n_matches = s.scalar(select(func.count()).select_from(RawMatch)) or 0
        n_elo = s.scalar(select(func.count()).select_from(RawEloSnapshot)) or 0
    return {"raw_matches": int(n_matches), "raw_elo_snapshots": int(n_elo)}


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
