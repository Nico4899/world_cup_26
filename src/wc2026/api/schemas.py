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


class BlendedOutcome(BaseModel):
    """Phase 5 blended-probability triplet, with provenance for both engines.

    Populated on ``PredictionResponse`` only when the optional XGB classifier
    is loaded and the caller requested a blend.
    """

    poisson: OutcomeProbabilities
    xgb: OutcomeProbabilities
    blended: OutcomeProbabilities
    weight: float = Field(description="Poisson mixing weight in [0, 1]; XGB gets 1 - weight.")


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
    blend: BlendedOutcome | None = Field(
        default=None,
        description=(
            "Phase 5 blended 1X2 from the geometric mean of PoissonDC and XGB. "
            "Populated only when the optional XGB classifier is loaded and the "
            "caller passed ``blend=true``."
        ),
    )


class FeatureContributionItem(BaseModel):
    """One feature's SHAP contribution toward a class prediction."""

    feature: str
    value: float | None = Field(
        default=None,
        description="The feature value at the predicted match (None if NaN/missing).",
    )
    contribution: float = Field(
        description="Signed SHAP value; positive pushes probability up, negative down."
    )


class MatchExplanation(BaseModel):
    """Top-N SHAP feature contributions for one match's predicted class."""

    home_team: str
    away_team: str
    match_date: date
    class_name: str = Field(description="The class being explained: home_win / draw / away_win.")
    contributions: list[FeatureContributionItem]
    poisson_outcome: OutcomeProbabilities
    xgb_outcome: OutcomeProbabilities | None = None


class LiveStateSnapshot(BaseModel):
    """Phase 6 live-match state + in-running win probabilities."""

    match_id: int
    home_team: str
    away_team: str
    minute: int
    period: int
    home_score: int
    away_score: int
    home_red_cards: int
    away_red_cards: int
    last_event_type: str = Field(
        description="Type of the most recent observed event: KICKOFF / GOAL / FT_WHISTLE."
    )
    win_prob: OutcomeProbabilities
    win_prob_source: str = Field(
        description=(
            "Where the win-prob came from: 'live_win_prob' (Phase 6 in-running model), "
            "'poisson_pre_match' (no events yet or live model unavailable), or "
            "'final' (match finished — the probability degenerates onto the realised outcome)."
        )
    )


class TeamAssetsResponse(BaseModel):
    """TheSportsDB-sourced UI assets for one team. ``null`` fields mean the
    upstream record exists but the field was blank."""

    team: str
    crest_url: str | None = None
    kit_home_color: str | None = None
    kit_away_color: str | None = None
    stadium_name: str | None = None
    stadium_capacity: int | None = None
    stadium_city: str | None = None
    stadium_country: str | None = None


class LiveEventTrace(BaseModel):
    """One in-running observation in the per-match timeline."""

    seq: int
    minute: int
    period: int
    event_type: str
    team: str | None = None
    home_score_after: int
    away_score_after: int
    home_red_cards_after: int
    away_red_cards_after: int
    win_prob: OutcomeProbabilities


class LiveHistory(BaseModel):
    """Snapshot + chronological list of every observed event for one fixture."""

    snapshot: LiveStateSnapshot
    events: list[LiveEventTrace]


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
