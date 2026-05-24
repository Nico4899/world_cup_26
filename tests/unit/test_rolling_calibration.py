"""Unit tests for the rolling WC 2026 calibration module."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from wc2026.db.models import Base, ModelPrediction, RawLiveEvent
from wc2026.eval.rolling import (
    compute_rolling,
    compute_rolling_from_dfs,
    load_completed_matches,
)


@pytest.fixture
def sqlite_engine():
    eng = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(eng)
    try:
        yield eng
    finally:
        eng.dispose()


def _pred_row(
    *,
    match_date: date,
    home: str,
    away: str,
    p_h: float = 0.5,
    p_d: float = 0.25,
    p_a: float = 0.25,
    created_at: datetime | None = None,
    model_version: str = "poisson_dc.v1",
) -> dict:
    return {
        "match_date": match_date,
        "home_team": home,
        "away_team": away,
        "p_home": p_h,
        "p_draw": p_d,
        "p_away": p_a,
        "model_version": model_version,
        "created_at": created_at or datetime(2026, 6, 1, tzinfo=UTC),
    }


def _completed_row(
    *,
    match_date: date,
    home: str,
    away: str,
    home_score: int,
    away_score: int,
) -> dict:
    observed = "H" if home_score > away_score else ("A" if away_score > home_score else "D")
    return {
        "match_date": match_date,
        "home_team": home,
        "away_team": away,
        "home_score": home_score,
        "away_score": away_score,
        "observed": observed,
    }


def test_compute_rolling_returns_zero_when_no_completed_matches() -> None:
    result = compute_rolling_from_dfs(pd.DataFrame(), pd.DataFrame())
    assert result.n_completed == 0
    assert result.log_loss is None
    assert result.brier is None
    assert result.per_match == []


def test_compute_rolling_picks_latest_pre_match_prediction() -> None:
    """Among predictions with created_at ≤ match-date midnight, we use the most recent."""
    match_date = date(2026, 6, 11)
    older = _pred_row(
        match_date=match_date,
        home="Mexico",
        away="Senegal",
        p_h=0.4,
        p_d=0.3,
        p_a=0.3,
        created_at=datetime(2026, 5, 23, tzinfo=UTC),
    )
    newer = _pred_row(
        match_date=match_date,
        home="Mexico",
        away="Senegal",
        p_h=0.55,
        p_d=0.25,
        p_a=0.2,
        created_at=datetime(2026, 6, 10, tzinfo=UTC),
    )
    preds = pd.DataFrame([older, newer])
    completed = pd.DataFrame(
        [_completed_row(match_date=match_date, home="Mexico", away="Senegal", home_score=2, away_score=0)]
    )
    result = compute_rolling_from_dfs(preds, completed)
    assert result.n_completed == 1
    assert result.per_match[0].p_home == 0.55  # picked the newer pre-match prediction


def test_compute_rolling_excludes_predictions_made_after_kickoff() -> None:
    """A prediction created after match-day midnight UTC must NOT count.

    This guards against the daily refit firing post-match (e.g. evening UTC)
    and accidentally writing a leakage-tainted snapshot.
    """
    match_date = date(2026, 6, 11)
    pre = _pred_row(
        match_date=match_date,
        home="Mexico",
        away="Senegal",
        p_h=0.45,
        created_at=datetime(2026, 6, 10, 23, 59, tzinfo=UTC),
    )
    post = _pred_row(
        match_date=match_date,
        home="Mexico",
        away="Senegal",
        p_h=0.99,
        created_at=datetime(2026, 6, 11, 12, 0, tzinfo=UTC),
    )
    preds = pd.DataFrame([pre, post])
    completed = pd.DataFrame(
        [_completed_row(match_date=match_date, home="Mexico", away="Senegal", home_score=1, away_score=0)]
    )
    result = compute_rolling_from_dfs(preds, completed)
    assert result.per_match[0].p_home == 0.45


def test_compute_rolling_computes_log_loss_brier_rps() -> None:
    """Single match with a known prediction → known calibration numbers."""
    import math

    match_date = date(2026, 6, 11)
    preds = pd.DataFrame(
        [
            _pred_row(
                match_date=match_date,
                home="Argentina",
                away="France",
                p_h=0.5,
                p_d=0.25,
                p_a=0.25,
                created_at=datetime(2026, 6, 10, tzinfo=UTC),
            )
        ]
    )
    completed = pd.DataFrame(
        [_completed_row(match_date=match_date, home="Argentina", away="France", home_score=2, away_score=1)]
    )
    result = compute_rolling_from_dfs(preds, completed)
    assert result.n_completed == 1
    # log_loss of a 0.5 home-win prediction is -log(0.5) = 0.693.
    assert math.isclose(result.log_loss, -math.log(0.5), abs_tol=1e-9)


def test_compute_rolling_aggregates_over_multiple_completed_matches() -> None:
    """Mean of per-match scores over multiple completed matches."""
    match_a = date(2026, 6, 11)
    match_b = date(2026, 6, 12)
    preds = pd.DataFrame(
        [
            _pred_row(
                match_date=match_a, home="Mexico", away="Senegal",
                p_h=0.6, p_d=0.25, p_a=0.15,
                created_at=datetime(2026, 6, 10, tzinfo=UTC),
            ),
            _pred_row(
                match_date=match_b, home="Argentina", away="France",
                p_h=0.4, p_d=0.30, p_a=0.30,
                created_at=datetime(2026, 6, 11, tzinfo=UTC),
            ),
        ]
    )
    completed = pd.DataFrame(
        [
            _completed_row(match_date=match_a, home="Mexico", away="Senegal", home_score=2, away_score=0),
            _completed_row(match_date=match_b, home="Argentina", away="France", home_score=0, away_score=1),
        ]
    )
    result = compute_rolling_from_dfs(preds, completed)
    assert result.n_completed == 2
    assert len(result.per_match) == 2


def test_compute_rolling_skips_matches_without_predictions() -> None:
    """A completed match without any pre-match prediction is silently skipped."""
    match_date = date(2026, 6, 11)
    completed = pd.DataFrame(
        [_completed_row(match_date=match_date, home="Mexico", away="Senegal", home_score=1, away_score=0)]
    )
    result = compute_rolling_from_dfs(pd.DataFrame(), completed)
    assert result.n_completed == 0


# --- DB-backed paths -------------------------------------------------------


def test_load_completed_matches_returns_empty_for_empty_mapping(sqlite_engine) -> None:
    df = load_completed_matches(sqlite_engine, match_id_to_fixture={})
    assert df.empty


def test_load_completed_matches_joins_via_match_id_mapping(sqlite_engine) -> None:
    """``RawLiveEvent.match_id`` is football-data.org's int; we map it back to
    our (date, home, away) natural key via the dict the caller supplies."""
    match_date = date(2026, 6, 11)
    with Session(sqlite_engine, future=True) as session:
        session.add(
            RawLiveEvent(
                match_id=12345,
                seq=4,
                minute=90,
                period=2,
                event_type="FT_WHISTLE",
                team=None,
                player=None,
                home_score_after=2,
                away_score_after=1,
                home_red_cards_after=0,
                away_red_cards_after=0,
                ingested_at=datetime.now(UTC),
            )
        )
        # A non-FT_WHISTLE row that should NOT be returned.
        session.add(
            RawLiveEvent(
                match_id=12345,
                seq=2,
                minute=23,
                period=1,
                event_type="GOAL",
                team="Mexico",
                player=None,
                home_score_after=1,
                away_score_after=0,
                home_red_cards_after=0,
                away_red_cards_after=0,
                ingested_at=datetime.now(UTC),
            )
        )
        session.commit()
    df = load_completed_matches(
        sqlite_engine, match_id_to_fixture={12345: (match_date, "Mexico", "Senegal")}
    )
    assert len(df) == 1
    assert df.iloc[0]["home_score"] == 2
    assert df.iloc[0]["observed"] == "H"


def test_compute_rolling_db_backed_end_to_end(sqlite_engine) -> None:
    """End-to-end with model_predictions + raw_live_events on SQLite."""
    match_date = date(2026, 6, 11)
    fdo_id = 99
    with Session(sqlite_engine, future=True) as session:
        session.add(
            ModelPrediction(
                match_date=match_date,
                home_team="Mexico",
                away_team="Senegal",
                p_home=0.6,
                p_draw=0.25,
                p_away=0.15,
                model_version="poisson_dc.v1",
                created_at=datetime(2026, 6, 10, tzinfo=UTC),
            )
        )
        session.add(
            RawLiveEvent(
                match_id=fdo_id,
                seq=4,
                minute=90,
                period=2,
                event_type="FT_WHISTLE",
                team=None,
                player=None,
                home_score_after=2,
                away_score_after=0,
                home_red_cards_after=0,
                away_red_cards_after=0,
                ingested_at=datetime.now(UTC),
            )
        )
        session.commit()
    result = compute_rolling(
        match_id_to_fixture={fdo_id: (match_date, "Mexico", "Senegal")},
        engine=sqlite_engine,
    )
    assert result.n_completed == 1
    assert result.per_match[0].observed == "H"


def test_compute_rolling_db_returns_zero_when_no_completed_matches(sqlite_engine) -> None:
    result = compute_rolling(match_id_to_fixture={}, engine=sqlite_engine)
    assert result.n_completed == 0
    assert result.log_loss is None
