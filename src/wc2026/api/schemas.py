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


class EloHistoryPoint(BaseModel):
    snapshot_date: date
    rating: float
    global_rank: int | None = None


class TeamEloHistory(BaseModel):
    team: str
    history: list[EloHistoryPoint]


class TeamTournamentProbabilities(BaseModel):
    """Per-team advancement probabilities pulled from the latest persisted MC run."""

    team: str
    run_id: int | None = None
    n_sims: int | None = None
    model_version: str | None = None
    group_winner_p: float | None = None
    group_runner_up_p: float | None = None
    advance_r32_p: float | None = None
    advance_r16_p: float | None = None
    quarterfinal_p: float | None = None
    semifinal_p: float | None = None
    final_p: float | None = None
    champion_p: float | None = None


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


class FifaRankingPoint(BaseModel):
    """One row of FIFA Men's Ranking history for a single team."""

    ranking_date: date
    rank: int
    points: float | None = None
    previous_rank: int | None = None


class TeamFifaRankingHistory(BaseModel):
    """Chronological FIFA ranking history (oldest → newest) for one team."""

    team: str
    history: list[FifaRankingPoint]


class SquadMember(BaseModel):
    """One player on a tournament squad snapshot."""

    player_name: str
    shirt_number: int | None = None
    position: str | None = None
    birth_date: date | None = None
    club: str | None = None
    caps: int | None = None
    goals: int | None = None


class TeamSquadResponse(BaseModel):
    """Latest tournament-squad snapshot for one team."""

    team: str
    tournament: str | None = None
    snapshot_date: date | None = None
    players: list[SquadMember] = []


class XgFormSplit(BaseModel):
    """Aggregate xG-for / xG-against over a rolling window."""

    matches: int
    xg_for: float | None = None
    xg_against: float | None = None
    xg_diff: float | None = None
    goals_for: int | None = None
    goals_against: int | None = None


class TeamXgFormResponse(BaseModel):
    """Last-N xG form, derived from the most recent ``raw_match_xg`` rows."""

    team: str
    last_5: XgFormSplit
    last_10: XgFormSplit
