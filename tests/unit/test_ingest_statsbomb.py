"""Unit tests for the StatsBomb open-data ingester."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from wc2026.ingest.statsbomb_open import (
    GOAL_X,
    GOAL_Y,
    _distance_and_angle,
    aggregate_match_xg,
    fetch_competition_shots,
    load_fixture_events,
    parse_events_shots,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


def _events() -> list[dict[str, Any]]:
    return load_fixture_events(FIXTURE_DIR / "statsbomb_events_sample.json")


def test_distance_zero_at_goal_centre() -> None:
    dist, _ = _distance_and_angle(GOAL_X, GOAL_Y)
    assert dist == 0.0


def test_distance_monotonic_with_x() -> None:
    """Moving along the central axis: distance to goal should fall as x→120."""
    d_near, _ = _distance_and_angle(110.0, 40.0)
    d_far, _ = _distance_and_angle(80.0, 40.0)
    assert d_near < d_far


def test_angle_wider_when_central_than_wide() -> None:
    """At the same distance, a central shot has a wider angle than a wide one."""
    _, a_centre = _distance_and_angle(108.0, 40.0)
    _, a_wide = _distance_and_angle(108.0, 5.0)
    assert a_centre > a_wide
    # Sanity: angle is in (0, pi).
    assert 0.0 < a_centre < math.pi
    assert 0.0 < a_wide < math.pi


def test_parse_events_filters_to_shots_only() -> None:
    df = parse_events_shots(
        _events(),
        match_id=1,
        match_date="2022-12-18",
        competition_id=43,
        season_id=106,
        home_team="Argentina",
        away_team="France",
    )
    # 5 events in fixture; 1 Pass, 4 Shots
    assert len(df) == 4
    assert df["team"].tolist() == ["Argentina", "Argentina", "France", "Argentina"]


def test_parse_events_populates_distance_angle_xg_outcome() -> None:
    df = parse_events_shots(
        _events(),
        match_id=1,
        match_date="2022-12-18",
        competition_id=43,
        season_id=106,
        home_team="Argentina",
        away_team="France",
    )
    # The first shot (Messi @108,35) was a goal with xG 0.27.
    first = df.iloc[0]
    assert first["statsbomb_xg"] == 0.27
    assert bool(first["is_goal"]) is True
    assert bool(first["is_penalty"]) is False
    assert first["distance_to_goal"] > 0
    assert first["angle_to_goal"] > 0


def test_parse_events_identifies_penalty_and_header() -> None:
    df = parse_events_shots(
        _events(),
        match_id=1,
        match_date="2022-12-18",
        competition_id=43,
        season_id=106,
        home_team="Argentina",
        away_team="France",
    )
    # The Messi 117' shot is a penalty.
    penalty = df[df["pattern_of_play"] == "Penalty"]
    assert len(penalty) == 1
    assert bool(penalty.iloc[0]["is_penalty"]) is True
    # No headers in this fixture.
    assert df["is_header"].sum() == 0


def test_parse_events_resolves_opponent_via_match_teams() -> None:
    df = parse_events_shots(
        _events(),
        match_id=1,
        match_date="2022-12-18",
        competition_id=43,
        season_id=106,
        home_team="Argentina",
        away_team="France",
    )
    arg = df[df["team"] == "Argentina"]
    fra = df[df["team"] == "France"]
    assert (arg["opponent"] == "France").all()
    assert (fra["opponent"] == "Argentina").all()


def test_parse_events_returns_empty_df_when_no_shots() -> None:
    df = parse_events_shots(
        [{"type": {"name": "Pass"}, "location": [0, 0]}],
        match_id=1,
        match_date="2022-12-18",
        competition_id=43,
        season_id=106,
        home_team="A",
        away_team="B",
    )
    assert df.empty


def test_aggregate_match_xg_sums_per_team() -> None:
    shots = parse_events_shots(
        _events(),
        match_id=1,
        match_date="2022-12-18",
        competition_id=43,
        season_id=106,
        home_team="Argentina",
        away_team="France",
    )
    agg = aggregate_match_xg(shots)
    # Two rows: one for Argentina, one for France.
    assert len(agg) == 2
    by_team = agg.set_index("team")
    assert math.isclose(by_team.loc["Argentina", "xg_for"], 0.27 + 0.55 + 0.78, abs_tol=1e-9)
    assert math.isclose(by_team.loc["France", "xg_for"], 0.18, abs_tol=1e-9)
    # xG against is the symmetric counterpart.
    assert math.isclose(by_team.loc["Argentina", "xg_against"], 0.18, abs_tol=1e-9)


def test_aggregate_match_xg_counts_shots() -> None:
    shots = parse_events_shots(
        _events(),
        match_id=1,
        match_date="2022-12-18",
        competition_id=43,
        season_id=106,
        home_team="Argentina",
        away_team="France",
    )
    agg = aggregate_match_xg(shots)
    by_team = agg.set_index("team")
    assert int(by_team.loc["Argentina", "shots"]) == 3
    assert int(by_team.loc["France", "shots"]) == 1


def test_aggregate_match_xg_empty_input_returns_empty_with_columns() -> None:
    agg = aggregate_match_xg(pd.DataFrame())
    assert agg.empty
    assert set(agg.columns) >= {"match_date", "team", "xg_for", "xg_against"}


# ----- fetcher with stubbed HTTP -------------------------------------------


class _StubResponse:
    def __init__(self, payload: Any):
        self._payload = payload
        self.status_code = 200

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _StubSession:
    def __init__(self, responses: dict[str, Any]):
        self._responses = responses
        self.headers: dict[str, str] = {}
        self.calls: list[str] = []

    def get(self, url: str, **_):
        self.calls.append(url)
        return _StubResponse(self._responses[url])


def test_fetch_competition_shots_writes_parquet_under_comp_season(tmp_path: Path) -> None:
    matches_payload = [
        {
            "match_id": 1,
            "match_date": "2022-12-18",
            "home_team": {"home_team_name": "Argentina"},
            "away_team": {"away_team_name": "France"},
        }
    ]
    responses = {
        "https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/43/106.json": matches_payload,
        "https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/1.json": _events(),
    }
    session = _StubSession(responses)
    out = fetch_competition_shots(43, 106, session=session, target_dir=tmp_path)
    assert out == tmp_path / "43" / "106" / "shots.parquet"
    df = pd.read_parquet(out)
    assert (df["match_id"] == 1).all()
    assert (df["competition_id"] == 43).all()
    assert (df["season_id"] == 106).all()
    assert len(df) == 4  # four shots in the fixture
