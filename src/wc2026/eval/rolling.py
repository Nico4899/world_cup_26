"""Rolling WC 2026 calibration over completed matches.

Reads from two tables already populated by other Phase 6/7 components:

* ``model_predictions``: one row per (match, model_version, created_at). We
  pick the row with the latest ``created_at <= match_date`` for each fixture —
  i.e. the *pre-match* prediction-of-record, never a post-hoc snapshot.
* ``raw_live_events``: filtered to ``FT_WHISTLE`` rows for completed matches'
  final scores.

Joins on ``(match_date, home_team, away_team)`` and rolls up Brier / log-loss
/ RPS using the same per-match scorers in ``eval.calibration`` that the
historical hindcasts use, so the live-tournament panel uses an apples-to-
apples comparison against the WC 2018 + WC 2022 baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
from sqlalchemy import desc, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from wc2026.db.models import ModelPrediction, RawLiveEvent
from wc2026.db.session import get_engine
from wc2026.eval.calibration import match_brier, match_log_loss, match_rps
from wc2026.ingest.live_events import EVENT_FT_WHISTLE


@dataclass(frozen=True)
class PerMatchCalibration:
    """One completed-match diagnostic row."""

    match_date: date
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    p_home: float
    p_draw: float
    p_away: float
    observed: str  # "H" / "D" / "A"
    log_loss: float
    brier: float
    rps: float
    model_version: str


@dataclass(frozen=True)
class RollingCalibration:
    """Aggregate WC 2026 calibration + per-match breakdown."""

    n_completed: int
    log_loss: float | None
    brier: float | None
    rps: float | None
    per_match: list[PerMatchCalibration]


def _observed_outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def load_completed_matches(
    engine: Engine,
    *,
    match_id_to_fixture: dict[int, tuple[date, str, str]] | None = None,
) -> pd.DataFrame:
    """Pull the FT_WHISTLE row for every completed match.

    Returns columns: ``match_date, home_team, away_team, home_score,
    away_score, observed``.

    The poller writes football-data.org's numeric match_id (a separate
    namespace from our (date, home, away) natural key) — so callers must
    pass ``match_id_to_fixture`` mapping it back. When the dict is empty,
    we return an empty DataFrame.
    """
    if not match_id_to_fixture:
        return pd.DataFrame(
            columns=[
                "match_date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "observed",
            ]
        )
    with Session(engine, future=True) as session:
        rows = list(
            session.scalars(select(RawLiveEvent).where(RawLiveEvent.event_type == EVENT_FT_WHISTLE))
        )
    out_rows: list[dict[str, Any]] = []
    for r in rows:
        fixture = match_id_to_fixture.get(r.match_id)
        if fixture is None:
            continue
        match_date, home_team, away_team = fixture
        out_rows.append(
            {
                "match_date": match_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": int(r.home_score_after),
                "away_score": int(r.away_score_after),
                "observed": _observed_outcome(int(r.home_score_after), int(r.away_score_after)),
            }
        )
    return pd.DataFrame(out_rows)


def _latest_pre_match_prediction(
    session: Session, *, match_date: date, home_team: str, away_team: str
) -> ModelPrediction | None:
    stmt = (
        select(ModelPrediction)
        .where(
            ModelPrediction.match_date == match_date,
            ModelPrediction.home_team == home_team,
            ModelPrediction.away_team == away_team,
            ModelPrediction.created_at <= _start_of_match_day(match_date),
        )
        .order_by(desc(ModelPrediction.created_at))
        .limit(1)
    )
    return session.scalars(stmt).first()


def _start_of_match_day(match_date: date):
    """Cut-off for `pre-match`: midnight UTC at the match date.

    A prediction persisted on the morning of the match qualifies, but one
    persisted *after* the match starts (e.g. by a refit job that runs the
    same day) does not.
    """
    return datetime(match_date.year, match_date.month, match_date.day, tzinfo=UTC)


def compute_rolling_from_dfs(
    predictions: pd.DataFrame,
    completed: pd.DataFrame,
) -> RollingCalibration:
    """Pure-function variant. Useful for tests that don't want to spin up SQL.

    ``predictions`` must contain columns:
        match_date, home_team, away_team, p_home, p_draw, p_away,
        model_version, created_at

    ``completed`` must contain columns:
        match_date, home_team, away_team, home_score, away_score, observed
    """
    if predictions.empty or completed.empty:
        return RollingCalibration(
            n_completed=0,
            log_loss=None,
            brier=None,
            rps=None,
            per_match=[],
        )
    # For each completed match, pick the most-recent prediction with
    # created_at strictly before the match date (UTC).
    preds = predictions.copy()
    preds["created_at"] = pd.to_datetime(preds["created_at"], utc=True)
    preds["match_date"] = pd.to_datetime(preds["match_date"]).dt.date
    completed = completed.copy()
    completed["match_date"] = pd.to_datetime(completed["match_date"]).dt.date

    rows: list[PerMatchCalibration] = []
    log_losses: list[float] = []
    briers: list[float] = []
    rpses: list[float] = []
    for _, fixture in completed.iterrows():
        cutoff = pd.Timestamp(fixture["match_date"], tz="UTC")
        candidates = preds[
            (preds["match_date"] == fixture["match_date"])
            & (preds["home_team"] == fixture["home_team"])
            & (preds["away_team"] == fixture["away_team"])
            & (preds["created_at"] <= cutoff)
        ]
        if candidates.empty:
            continue
        latest = candidates.sort_values("created_at", ascending=False).iloc[0]
        observed = fixture["observed"]
        p_h = float(latest["p_home"])
        p_d = float(latest["p_draw"])
        p_a = float(latest["p_away"])
        ll = match_log_loss(observed, p_h, p_d, p_a)
        br = match_brier(observed, p_h, p_d, p_a)
        rp = match_rps(observed, p_h, p_d, p_a)
        log_losses.append(ll)
        briers.append(br)
        rpses.append(rp)
        rows.append(
            PerMatchCalibration(
                match_date=fixture["match_date"],
                home_team=fixture["home_team"],
                away_team=fixture["away_team"],
                home_score=int(fixture["home_score"]),
                away_score=int(fixture["away_score"]),
                p_home=p_h,
                p_draw=p_d,
                p_away=p_a,
                observed=observed,
                log_loss=ll,
                brier=br,
                rps=rp,
                model_version=str(latest["model_version"]),
            )
        )
    if not rows:
        return RollingCalibration(
            n_completed=0,
            log_loss=None,
            brier=None,
            rps=None,
            per_match=[],
        )
    n = len(rows)
    return RollingCalibration(
        n_completed=n,
        log_loss=sum(log_losses) / n,
        brier=sum(briers) / n,
        rps=sum(rpses) / n,
        per_match=rows,
    )


def compute_rolling(
    *,
    match_id_to_fixture: dict[int, tuple[date, str, str]],
    engine: Engine | None = None,
) -> RollingCalibration:
    """DB-backed variant. Returns the aggregate + per-match diagnostics."""
    eng = engine or get_engine()
    completed = load_completed_matches(eng, match_id_to_fixture=match_id_to_fixture)
    if completed.empty:
        return RollingCalibration(n_completed=0, log_loss=None, brier=None, rps=None, per_match=[])
    with Session(eng, future=True) as session:
        # Cheap: pull every WC 2026 prediction row in one query. Even with 14
        # daily snapshots per fixture x 72 fixtures the result fits in memory.
        match_dates = [c[0] for c in match_id_to_fixture.values()]
        stmt = select(ModelPrediction).where(ModelPrediction.match_date.in_(match_dates))
        preds = pd.DataFrame(
            [
                {
                    "match_date": row.match_date,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "p_home": row.p_home,
                    "p_draw": row.p_draw,
                    "p_away": row.p_away,
                    "model_version": row.model_version,
                    "created_at": row.created_at,
                }
                for row in session.scalars(stmt)
            ]
        )
    return compute_rolling_from_dfs(preds, completed)


__all__ = [
    "PerMatchCalibration",
    "RollingCalibration",
    "compute_rolling",
    "compute_rolling_from_dfs",
    "load_completed_matches",
]
