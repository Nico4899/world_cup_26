"""SQLAlchemy 2.0 declarative models for the wc2026 Postgres warehouse.

These tables hold raw ingest output, model predictions, and bookkeeping for the
scheduler. The application code never writes to these tables directly except via
the load scripts and scheduler jobs; the read paths in `wc2026.api` consume them.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Common declarative base for all wc2026 tables."""


class RawMatch(Base):
    __tablename__ = "raw_matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    home_team: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    away_team: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    home_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tournament: Mapped[str] = mapped_column(String(128), nullable=False)
    city: Mapped[str | None] = mapped_column(String(128), nullable=True)
    country: Mapped[str | None] = mapped_column(String(128), nullable=True)
    neutral: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    __table_args__ = (
        UniqueConstraint(
            "date", "home_team", "away_team", "source", name="uq_raw_matches_natural_key"
        ),
    )


class RawEloSnapshot(Base):
    __tablename__ = "raw_elo_snapshots"

    snapshot_date: Mapped[date] = mapped_column(Date, primary_key=True)
    team_code: Mapped[str] = mapped_column(String(8), primary_key=True)
    team_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    rating: Mapped[float] = mapped_column(Float, nullable=False)
    global_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)


class RawEloOverride(Base):
    """Operator-set Elo override for a single team.

    Loaded on top of the disk-side ``elo_current_*.parquet`` snapshot when
    the eloratings.net scraper is broken or returns obviously-wrong values.
    The override is keyed by ``team_code`` and survives until cleared via
    ``DELETE /api/v1/_ops/elo-override/{team_code}``. There is intentionally
    no expiry — the operator that sets it is the one who clears it.
    """

    __tablename__ = "raw_elo_overrides"

    team_code: Mapped[str] = mapped_column(String(8), primary_key=True)
    team_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    rating: Mapped[float] = mapped_column(Float, nullable=False)
    reason: Mapped[str | None] = mapped_column(String(256), nullable=True)
    set_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class ModelPrediction(Base):
    __tablename__ = "model_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    home_team: Mapped[str] = mapped_column(String(128), nullable=False)
    away_team: Mapped[str] = mapped_column(String(128), nullable=False)
    p_home: Mapped[float] = mapped_column(Float, nullable=False)
    p_draw: Mapped[float] = mapped_column(Float, nullable=False)
    p_away: Mapped[float] = mapped_column(Float, nullable=False)
    score_matrix_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class TournamentSimRun(Base):
    __tablename__ = "tournament_sim_runs"

    run_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    n_sims: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)

    team_outcomes: Mapped[list[TournamentSimTeamOutcome]] = relationship(
        "TournamentSimTeamOutcome",
        back_populates="run",
        cascade="all, delete-orphan",
    )


class TournamentSimTeamOutcome(Base):
    __tablename__ = "tournament_sim_team_outcomes"

    run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("tournament_sim_runs.run_id", ondelete="CASCADE"),
        primary_key=True,
    )
    team: Mapped[str] = mapped_column(String(128), primary_key=True)
    group_winner_p: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    group_runner_up_p: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Phase 12: split eliminated mass into 3rd-place-but-not-best-8 (third_out_p)
    # and 4th-place (fourth_p). Nullable so pre-Phase-12 rows still load.
    third_out_p: Mapped[float | None] = mapped_column(Float, nullable=True)
    fourth_p: Mapped[float | None] = mapped_column(Float, nullable=True)
    advance_r32_p: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    advance_r16_p: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    quarterfinal_p: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    semifinal_p: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    final_p: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    champion_p: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    run: Mapped[TournamentSimRun] = relationship("TournamentSimRun", back_populates="team_outcomes")


class SchedulerJobRun(Base):
    __tablename__ = "scheduler_job_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)


class RawTeamAsset(Base):
    """TheSportsDB-sourced crest / kit / stadium metadata, one row per team.

    ``team`` is the canonical name as it appears in ``raw_matches.home_team`` —
    the ingester is responsible for resolving TheSportsDB's spelling back to it
    via a hand-maintained alias map.
    """

    __tablename__ = "raw_team_assets"

    team: Mapped[str] = mapped_column(String(128), primary_key=True)
    thesportsdb_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    crest_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    kit_home_color: Mapped[str | None] = mapped_column(String(16), nullable=True)
    kit_away_color: Mapped[str | None] = mapped_column(String(16), nullable=True)
    stadium_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    stadium_capacity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stadium_city: Mapped[str | None] = mapped_column(String(128), nullable=True)
    stadium_country: Mapped[str | None] = mapped_column(String(128), nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class RawSquad(Base):
    """Tournament squad row (one per player per snapshot_date).

    Squads change frequently in the months before a tournament; the
    (tournament, team, player_name, snapshot_date) PK lets us keep a history
    of how each roster evolved instead of a single mutable row.
    """

    __tablename__ = "raw_squads"

    tournament: Mapped[str] = mapped_column(String(128), primary_key=True)
    team: Mapped[str] = mapped_column(String(128), primary_key=True)
    player_name: Mapped[str] = mapped_column(String(128), primary_key=True)
    snapshot_date: Mapped[date] = mapped_column(Date, primary_key=True)
    shirt_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    position: Mapped[str | None] = mapped_column(String(8), nullable=True)
    birth_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    club: Mapped[str | None] = mapped_column(String(128), nullable=True)
    caps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class RawFifaRanking(Base):
    """FIFA Men's Ranking row (one per (date, team))."""

    __tablename__ = "raw_fifa_rankings"

    ranking_date: Mapped[date] = mapped_column(Date, primary_key=True)
    team: Mapped[str] = mapped_column(String(128), primary_key=True)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    points: Mapped[float | None] = mapped_column(Float, nullable=True)
    previous_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class MatchFeatures(Base):
    """Materialised per-match feature row consumed by Phase 5's XGBoost model.

    One row per (home_team, away_team, match_date). Numeric features are NaN-
    tolerant: callers downstream decide how to impute missing values. The
    ``source_snapshots`` JSON column records *which* upstream snapshots fed
    this row so a regression can be re-pinned to the exact data state.
    """

    __tablename__ = "features_match_features"

    match_date: Mapped[date] = mapped_column(Date, primary_key=True)
    home_team: Mapped[str] = mapped_column(String(128), primary_key=True)
    away_team: Mapped[str] = mapped_column(String(128), primary_key=True)

    elo_diff: Mapped[float | None] = mapped_column(Float, nullable=True)
    fifa_rank_diff: Mapped[float | None] = mapped_column(Float, nullable=True)
    xg_form_diff: Mapped[float | None] = mapped_column(Float, nullable=True)
    rest_days_diff: Mapped[float | None] = mapped_column(Float, nullable=True)
    squad_age_diff: Mapped[float | None] = mapped_column(Float, nullable=True)

    is_neutral: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_host_home: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_host_away: Mapped[int | None] = mapped_column(Integer, nullable=True)

    poisson_exp_home_goals: Mapped[float | None] = mapped_column(Float, nullable=True)
    poisson_exp_away_goals: Mapped[float | None] = mapped_column(Float, nullable=True)
    poisson_p_home: Mapped[float | None] = mapped_column(Float, nullable=True)
    poisson_p_draw: Mapped[float | None] = mapped_column(Float, nullable=True)
    poisson_p_away: Mapped[float | None] = mapped_column(Float, nullable=True)

    source_snapshots: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    built_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class RawLiveEvent(Base):
    """In-match event from a live football-data.org / TheSportsDB poll.

    One row per **observed** state-changing event (goal, red/yellow card,
    substitution, period-end). The ``seq`` column is the order the poller
    saw the event for this match; combined with ``match_id`` it forms the
    primary key. ``home_score_after`` / ``away_score_after`` /
    ``home_red_cards_after`` / ``away_red_cards_after`` capture the match
    state *after* this event resolves — so the row by itself contains every
    feature the live win-prob model needs.
    """

    __tablename__ = "raw_live_events"

    match_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    seq: Mapped[int] = mapped_column(Integer, primary_key=True)
    minute: Mapped[int] = mapped_column(Integer, nullable=False)
    period: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    team: Mapped[str | None] = mapped_column(String(128), nullable=True)
    player: Mapped[str | None] = mapped_column(String(128), nullable=True)
    home_score_after: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    away_score_after: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    home_red_cards_after: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    away_red_cards_after: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class RawMatchXg(Base):
    """Per-match, per-team expected-goals aggregate.

    The shot-level StatsBomb events live as Parquet snapshots under
    ``data/raw/statsbomb/`` (too many rows to hold in Postgres efficiently).
    This table is the **summary** used by feature engineering — one row per
    ``(match_date, home_team, away_team, team, source)`` so we can blend
    multiple xG sources (StatsBomb open data vs FBref) without overwriting
    each other.
    """

    __tablename__ = "raw_match_xg"

    match_date: Mapped[date] = mapped_column(Date, primary_key=True)
    home_team: Mapped[str] = mapped_column(String(128), primary_key=True)
    away_team: Mapped[str] = mapped_column(String(128), primary_key=True)
    team: Mapped[str] = mapped_column(String(128), primary_key=True)
    source: Mapped[str] = mapped_column(String(32), primary_key=True)
    xg_for: Mapped[float] = mapped_column(Float, nullable=False)
    xg_against: Mapped[float] = mapped_column(Float, nullable=False)
    shots: Mapped[int | None] = mapped_column(Integer, nullable=True)
    shots_on_target: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


__all__ = [
    "Base",
    "MatchFeatures",
    "ModelPrediction",
    "RawEloOverride",
    "RawEloSnapshot",
    "RawFifaRanking",
    "RawLiveEvent",
    "RawMatch",
    "RawMatchXg",
    "RawSquad",
    "RawTeamAsset",
    "SchedulerJobRun",
    "TournamentSimRun",
    "TournamentSimTeamOutcome",
]
