"""Pydantic response schemas for the FastAPI app."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_fitted: bool
    model_teams_n: int


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
