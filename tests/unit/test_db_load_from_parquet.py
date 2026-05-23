"""Unit tests for the parquet-loader helpers (no DB access).

The full upsert path needs Postgres, but the row-shaping and chunking helpers
are pure functions over pandas/Python data and can be exercised offline.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
from scripts import db_load_from_parquet as loader


def test_chunks_partitions_payload_into_expected_sized_lists():
    payload = [{"i": i} for i in range(12)]
    chunks = list(loader._chunks(payload, size=5))
    assert [len(c) for c in chunks] == [5, 5, 2]
    assert chunks[0][0] == {"i": 0}
    assert chunks[-1][-1] == {"i": 11}


def test_chunks_empty_payload_yields_nothing():
    assert list(loader._chunks([], size=10)) == []


def test_chunks_size_larger_than_payload_yields_single_chunk():
    payload = [{"i": 0}, {"i": 1}]
    assert list(loader._chunks(payload, size=100)) == [payload]


def test_row_to_match_payload_handles_nan_scores():
    df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-06-11"),
                "home_team": "Mexico",
                "away_team": "Senegal",
                "home_score": pd.NA,
                "away_score": pd.NA,
                "tournament": "FIFA World Cup",
                "city": pd.NA,
                "country": pd.NA,
                "neutral": False,
            }
        ]
    )
    now = datetime.now(UTC)
    row = next(df.itertuples(index=False))
    out = loader._row_to_match_payload(row, source="jurisoo_kaggle", now=now)
    assert out["home_score"] is None
    assert out["away_score"] is None
    assert out["city"] is None
    assert out["country"] is None
    assert out["date"] == date(2026, 6, 11)
    assert out["ingested_at"] is now


def test_row_to_match_payload_coerces_int_scores():
    df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-06-11"),
                "home_team": "Mexico",
                "away_team": "Senegal",
                "home_score": 2,
                "away_score": 1,
                "tournament": "FIFA World Cup",
                "city": "Mexico City",
                "country": "Mexico",
                "neutral": False,
            }
        ]
    )
    row = next(df.itertuples(index=False))
    out = loader._row_to_match_payload(row, source="jurisoo_kaggle", now=datetime.now(UTC))
    assert out["home_score"] == 2 and isinstance(out["home_score"], int)
    assert out["away_score"] == 1 and isinstance(out["away_score"], int)
    assert out["neutral"] is False


def test_row_to_elo_payload_handles_missing_team_name():
    df = pd.DataFrame(
        [
            {
                "snapshot_date": pd.Timestamp("2026-05-23"),
                "code": "ARG",
                "team_name": pd.NA,
                "rating": 2147.0,
                "global_rank": pd.NA,
            }
        ]
    )
    row = next(df.itertuples(index=False))
    out = loader._row_to_elo_payload(row)
    assert out["team_name"] is None
    assert out["global_rank"] is None
    assert out["rating"] == 2147.0
    assert out["snapshot_date"] == date(2026, 5, 23)
    assert out["team_code"] == "ARG"


def test_batch_size_constant_stays_under_pg_param_limit():
    assert loader.INSERT_BATCH_ROWS * 11 < 65535, "raw_matches batch would exceed PG limit"
    assert loader.INSERT_BATCH_ROWS * 5 < 65535, "raw_elo batch would exceed PG limit"
