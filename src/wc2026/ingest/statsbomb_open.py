"""Ingest event-level shot data from the StatsBomb open-data GitHub repo.

Source: https://github.com/statsbomb/open-data

We deliberately avoid the ``statsbombpy`` dependency and fetch the JSON files
directly via HTTPS — they live under known paths in the repo
(``data/competitions.json``, ``data/matches/{comp}/{season}.json``,
``data/events/{match_id}.json``). This keeps the dependency footprint flat
and reuses the same ``requests-cache`` session pattern as the other ingesters.

Why event-level
---------------
A per-shot model (logistic on shot location + body-part + pattern-of-play) is
the standard "shot-based xG" recipe (Caley 2015; Decroos et al. 2018) and is
what feeds Phase 5's ML layer.  We only need the *shot* events; everything
else (passes, dribbles, …) is discarded to keep the Parquet snapshot small.

Open competitions we care about (all men's senior):
    competition_id  season_id  name
    --------------  ---------  --------------------------------
    43              3          FIFA World Cup 2018
    43              106        FIFA World Cup 2022
    55              43         UEFA Euro 2020
    55              282        UEFA Euro 2024

The integer ids above match the StatsBomb open-data convention; the actual
list is discoverable via ``fetch_competitions()``.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import requests_cache

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMPETITIONS_URL = f"{BASE_URL}/competitions.json"

DEFAULT_TARGET = Path("data/raw/statsbomb")
DEFAULT_CACHE = Path("data/raw/statsbomb/.http_cache")
DEFAULT_CACHE_EXPIRY_SECONDS = 30 * 24 * 3600  # open data is effectively immutable

USER_AGENT = (
    "wc2026-predictor/0.1 "
    "(+https://github.com/Nico4899/world_cup_26; nico.fliegel@gmail.com) "
    "personal-research; calibrated WC 2026 predictions"
)

# StatsBomb pitch dimensions (https://github.com/statsbomb/open-data/blob/master/doc/StatsBomb%20Open%20Data%20Specification%20v1.1.pdf)
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_X = PITCH_LENGTH  # right-end goal centre (x=120, y=40)
GOAL_Y = PITCH_WIDTH / 2.0
GOAL_WIDTH = 8.0  # standard 8-yard goal — used for angle feature

# Curated list of men's competitions we care about. Override via the optional
# `competitions_filter` argument to fetch_all_shots.
MENS_TOURNAMENT_COMPETITIONS: tuple[tuple[int, int, str], ...] = (
    (43, 3, "FIFA World Cup 2018"),
    (43, 106, "FIFA World Cup 2022"),
    (55, 43, "UEFA Euro 2020"),
    (55, 282, "UEFA Euro 2024"),
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShotRow:
    """Schema documentation — actual DataFrame uses the same column names."""

    match_id: int
    match_date: str
    competition_id: int
    season_id: int
    minute: int
    period: int
    team: str
    opponent: str
    player: str | None
    x: float
    y: float
    distance_to_goal: float
    angle_to_goal: float
    body_part: str | None
    pattern_of_play: str | None  # shot.type.name: "Open Play" / "Free Kick" / ...
    technique: str | None
    statsbomb_xg: float | None
    is_goal: bool
    is_penalty: bool
    is_header: bool


def _make_session(
    cache_path: Path | None = DEFAULT_CACHE,
    expire_seconds: int = DEFAULT_CACHE_EXPIRY_SECONDS,
) -> requests.Session:
    if cache_path is None:
        session: requests.Session = requests.Session()
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        session = requests_cache.CachedSession(
            cache_name=str(cache_path),
            backend="sqlite",
            expire_after=expire_seconds,
            allowable_codes=(200,),
            stale_if_error=True,
        )
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return session


def _distance_and_angle(x: float, y: float) -> tuple[float, float]:
    """Distance to goal centre and shot angle (radians) given pitch coords.

    Distance is plain Euclidean to (120, 40). Angle uses the standard shot-on-
    goal formula: the angle subtended by the two posts as seen from the shot
    location. Both features are inputs to the xG shot model.
    """
    dx = GOAL_X - x
    dy = GOAL_Y - y
    distance = math.sqrt(dx * dx + dy * dy)
    # Angle subtended by the goal posts as seen from (x, y).
    left_post_y = GOAL_Y - GOAL_WIDTH / 2.0
    right_post_y = GOAL_Y + GOAL_WIDTH / 2.0
    # Treat dx==0 as on the goal-line: maximal angle straight in front.
    if dx <= 0:
        return distance, math.pi
    a_left = math.atan2(left_post_y - y, dx)
    a_right = math.atan2(right_post_y - y, dx)
    angle = abs(a_right - a_left)
    return distance, angle


def _get(d: dict[str, Any] | None, *keys: str) -> Any:
    """Safe nested ``.get`` over a JSON object."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def parse_events_shots(
    events: list[dict[str, Any]],
    *,
    match_id: int,
    match_date: str,
    competition_id: int,
    season_id: int,
    home_team: str,
    away_team: str,
) -> pd.DataFrame:
    """Filter ``events`` to shots and return a typed shot DataFrame.

    Skips events whose ``location`` is missing (shouldn't happen for shots,
    but the parser is defensive). ``team`` is the shooting team; ``opponent``
    is the conceding team (used for xg_against aggregates).
    """
    rows: list[dict[str, Any]] = []
    for ev in events:
        if _get(ev, "type", "name") != "Shot":
            continue
        loc = ev.get("location") or []
        if len(loc) < 2:
            continue
        x, y = float(loc[0]), float(loc[1])
        dist, angle = _distance_and_angle(x, y)
        shooting_team = _get(ev, "team", "name") or ""
        if shooting_team == home_team:
            opponent = away_team
        elif shooting_team == away_team:
            opponent = home_team
        else:
            # Defensive: an unknown team name (e.g., a typo or a typing mismatch
            # in the fixture); leave the opponent blank so the row is still usable.
            opponent = ""
        pattern = _get(ev, "shot", "type", "name")
        body_part = _get(ev, "shot", "body_part", "name")
        technique = _get(ev, "shot", "technique", "name")
        rows.append(
            {
                "match_id": match_id,
                "match_date": match_date,
                "competition_id": competition_id,
                "season_id": season_id,
                "minute": int(ev.get("minute") or 0),
                "period": int(ev.get("period") or 1),
                "team": shooting_team,
                "opponent": opponent,
                "player": _get(ev, "player", "name"),
                "x": x,
                "y": y,
                "distance_to_goal": dist,
                "angle_to_goal": angle,
                "body_part": body_part,
                "pattern_of_play": pattern,
                "technique": technique,
                "statsbomb_xg": _get(ev, "shot", "statsbomb_xg"),
                "is_goal": _get(ev, "shot", "outcome", "name") == "Goal",
                "is_penalty": pattern == "Penalty",
                "is_header": (body_part or "").lower() == "head",
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.astype(
        {
            "match_id": "Int64",
            "competition_id": "Int64",
            "season_id": "Int64",
            "minute": "Int64",
            "period": "Int64",
            "team": "string",
            "opponent": "string",
            "player": "string",
            "x": "float64",
            "y": "float64",
            "distance_to_goal": "float64",
            "angle_to_goal": "float64",
            "body_part": "string",
            "pattern_of_play": "string",
            "technique": "string",
            "statsbomb_xg": "float64",
            "is_goal": "bool",
            "is_penalty": "bool",
            "is_header": "bool",
        }
    )


def aggregate_match_xg(shots: pd.DataFrame) -> pd.DataFrame:
    """Turn per-shot rows into per-(match, team) xG_for / xG_against / shot counts.

    Returns a DataFrame with columns matching ``RawMatchXg``:
        match_date, home_team, away_team, team, xg_for, xg_against, shots, shots_on_target
    Uses the StatsBomb-provided ``statsbomb_xg``; rows with NaN xG are
    excluded from the sums.
    """
    if shots.empty:
        return pd.DataFrame(
            columns=[
                "match_date",
                "home_team",
                "away_team",
                "team",
                "xg_for",
                "xg_against",
                "shots",
                "shots_on_target",
            ]
        )
    out_rows: list[dict[str, Any]] = []
    for match_id, group in shots.groupby("match_id"):
        match_date = group["match_date"].iloc[0]
        teams = sorted({t for t in group["team"].dropna().unique() if t})
        if len(teams) != 2:
            # Defensive: missing/extra team names — skip rather than emit junk.
            continue
        home_team, away_team = teams[0], teams[1]
        for shooting_team, opponent in ((home_team, away_team), (away_team, home_team)):
            team_shots = group[group["team"] == shooting_team]
            xg_for = float(team_shots["statsbomb_xg"].dropna().sum())
            conceded = group[group["team"] == opponent]
            xg_against = float(conceded["statsbomb_xg"].dropna().sum())
            out_rows.append(
                {
                    "match_date": match_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "team": shooting_team,
                    "xg_for": xg_for,
                    "xg_against": xg_against,
                    "shots": len(team_shots),
                    "shots_on_target": int(team_shots["statsbomb_xg"].notna().sum()),
                }
            )
        _ = match_id  # silence unused-loop-variable; we use it via groupby
    return pd.DataFrame(out_rows)


def fetch_json(url: str, *, session: requests.Session) -> Any:
    """GET a JSON file, returning the parsed body."""
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_competitions(*, session: requests.Session | None = None) -> pd.DataFrame:
    sess = session or _make_session()
    payload = fetch_json(COMPETITIONS_URL, session=sess)
    return pd.DataFrame(payload)


def fetch_matches(
    competition_id: int,
    season_id: int,
    *,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    sess = session or _make_session()
    url = f"{BASE_URL}/matches/{competition_id}/{season_id}.json"
    return fetch_json(url, session=sess)


def fetch_match_events(
    match_id: int,
    *,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    sess = session or _make_session()
    url = f"{BASE_URL}/events/{match_id}.json"
    return fetch_json(url, session=sess)


def fetch_competition_shots(
    competition_id: int,
    season_id: int,
    *,
    session: requests.Session | None = None,
    target_dir: Path = DEFAULT_TARGET,
    match_ids: Iterable[int] | None = None,
) -> Path:
    """Fetch every shot event for one (competition, season) → Parquet snapshot.

    Writes to ``target_dir / {competition_id} / {season_id} / shots.parquet`` and
    returns that path. If ``match_ids`` is provided, only those matches are
    fetched (useful for smoke tests).
    """
    sess = session or _make_session()
    matches = fetch_matches(competition_id, season_id, session=sess)
    selected_ids = set(match_ids) if match_ids is not None else None
    frames: list[pd.DataFrame] = []
    for m in matches:
        mid = m.get("match_id")
        if mid is None:
            continue
        if selected_ids is not None and mid not in selected_ids:
            continue
        events = fetch_match_events(int(mid), session=sess)
        shots = parse_events_shots(
            events,
            match_id=int(mid),
            match_date=str(m.get("match_date") or ""),
            competition_id=competition_id,
            season_id=season_id,
            home_team=_get(m, "home_team", "home_team_name") or "",
            away_team=_get(m, "away_team", "away_team_name") or "",
        )
        if not shots.empty:
            frames.append(shots)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["match_id"])
    out_dir = target_dir / str(competition_id) / str(season_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "shots.parquet"
    df.to_parquet(out, index=False)
    return out


def fetch_all_tournament_shots(
    competitions: Iterable[tuple[int, int, str]] = MENS_TOURNAMENT_COMPETITIONS,
    *,
    session: requests.Session | None = None,
    target_dir: Path = DEFAULT_TARGET,
) -> list[Path]:
    """Fetch shots for every (competition, season, name) tuple. Returns the Parquet paths."""
    sess = session or _make_session()
    paths: list[Path] = []
    for comp_id, season_id, name in competitions:
        try:
            paths.append(
                fetch_competition_shots(comp_id, season_id, session=sess, target_dir=target_dir)
            )
        except requests.HTTPError as exc:
            logger.warning(
                "StatsBomb fetch for %s (%d/%d) failed: %s", name, comp_id, season_id, exc
            )
    return paths


def load_shots_corpus(target_dir: Path = DEFAULT_TARGET) -> pd.DataFrame:
    """Concatenate every ``shots.parquet`` in ``target_dir``.

    Returns an empty DataFrame (with the expected columns) if none exist.
    """
    paths = sorted(target_dir.glob("*/*/shots.parquet"))
    if not paths:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)


def load_fixture_events(path: Path) -> list[dict[str, Any]]:
    """Convenience: load a saved events.json off disk (used by tests)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "BASE_URL",
    "COMPETITIONS_URL",
    "DEFAULT_CACHE",
    "DEFAULT_TARGET",
    "GOAL_WIDTH",
    "GOAL_X",
    "GOAL_Y",
    "MENS_TOURNAMENT_COMPETITIONS",
    "PITCH_LENGTH",
    "PITCH_WIDTH",
    "ShotRow",
    "aggregate_match_xg",
    "fetch_all_tournament_shots",
    "fetch_competition_shots",
    "fetch_competitions",
    "fetch_match_events",
    "fetch_matches",
    "load_fixture_events",
    "load_shots_corpus",
    "parse_events_shots",
]
