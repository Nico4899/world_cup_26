"""Pydantic response schemas for the FastAPI app."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_fitted: bool
    model_teams_n: int
    model_fit_at: datetime | None = Field(
        default=None,
        description="UTC timestamp the in-memory model was fit (lifespan startup time).",
    )
    model_version: str | None = Field(
        default=None,
        description="Identifier of the currently-loaded model (e.g. 'poisson_dc.v1').",
    )
    elo_snapshot_date: date | None = Field(
        default=None,
        description="Date of the eloratings snapshot powering the shootout submodel; "
        "useful for spotting silent staleness if the daily scheduler stops running.",
    )
    elo_snapshot_age_days: int | None = Field(
        default=None,
        description="Days since the elo snapshot was captured. >7 means the daily ingest "
        "hasn't run recently — check the scheduler logs.",
    )
    shootout_model_loaded: bool = Field(
        default=False,
        description="True if the Elo-based shootout submodel was loaded at startup; "
        "False means knockouts fall back to 50/50.",
    )
    group_assignment_source: str = Field(
        default="derived",
        description=(
            "'derived' = group letters A-L came from fixture-date clique ordering; "
            "'official:<citation>' = letters came from the JSON file at "
            "data/wc2026_group_assignment.json (FIFA draw)."
        ),
    )


class FixtureSummary(BaseModel):
    """One scheduled WC 2026 group-stage match (used in list endpoints)."""

    match_id: int = Field(description="0-indexed position in the fixtures list (0..71)")
    date: date
    home_team: str
    away_team: str
    group: str
    city: str
    country: str
    neutral: bool


class Scoreline(BaseModel):
    home_goals: int
    away_goals: int
    probability: float


class OutcomeProbabilities(BaseModel):
    home_win: float
    draw: float
    away_win: float


class PredictionResponse(BaseModel):
    home_team: str
    away_team: str
    neutral: bool
    expected_home_goals: float
    expected_away_goals: float
    outcome: OutcomeProbabilities
    top_scorelines: list[Scoreline]
    score_matrix: list[list[float]] | None = Field(
        default=None,
        description=(
            "Full joint P(home_goals, away_goals) matrix. Rows = home goals 0..N, "
            "columns = away goals 0..N. Populated by /api/v1/predictions and "
            "/api/v1/matches/{id}; omitted (null) where not needed to keep responses small."
        ),
    )


class FixtureWithPrediction(BaseModel):
    fixture: FixtureSummary
    prediction: PredictionResponse


class TeamRecentMatch(BaseModel):
    """One past match, framed from a specific team's perspective."""

    date: date
    opponent: str
    venue: str = Field(description="home / away / neutral")
    goals_for: int
    goals_against: int
    result: str = Field(description="W / D / L from the team's perspective")
    tournament: str


class H2HMatch(BaseModel):
    """One head-to-head match between two specific teams (date-sorted desc)."""

    date: date
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    tournament: str
    neutral: bool
