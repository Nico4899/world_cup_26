"""Fit the Phase 6 in-match live win-probability model.

Pipeline:

1. Walk every shots-Parquet snapshot on disk (one per StatsBomb tournament
   the operator has run ``statsbomb_refresh`` for).
2. For each unique match, fetch the **full** events JSON via the cached
   ``ingest.statsbomb_open`` session, replay it through
   ``features.live_state.replay_statsbomb_events``, and emit one training
   row per state-changing event.
3. Tag each row with the pre-match Elo difference (from the latest Elo
   snapshot — a mild approximation since we use *current* Elo, not the
   historical Elo at match date) and the eventual full-time outcome.
4. Fit ``models.live_win_prob.LiveWinProbModel`` and persist.

CLI usage::

    uv run python scripts/fit_live_win_prob.py          # build corpus + fit + save
    uv run python scripts/fit_live_win_prob.py --rows N # only emit N rows (debug)

The script is also importable: ``fit_and_save(rows=...)`` lets callers pass
their own training rows (useful for tests and for tournaments not yet in the
StatsBomb open archive).
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wc2026.features.live_state import (
    replay_statsbomb_events,
    snapshots_to_training_rows,
)
from wc2026.ingest.eloratings_scraper import load_latest_snapshot as load_elo_snapshot
from wc2026.ingest.statsbomb_open import (
    DEFAULT_TARGET as STATSBOMB_DEFAULT_TARGET,
    fetch_match_events,
    fetch_matches,
)
from wc2026.models.live_win_prob import DEFAULT_ARTIFACT_PATH, LiveWinProbModel

logger = logging.getLogger(__name__)


def _elo_by_team(elo_df: pd.DataFrame | None) -> dict[str, float]:
    if elo_df is None or elo_df.empty:
        return {}
    if "team_name" not in elo_df.columns or "rating" not in elo_df.columns:
        return {}
    return {
        str(name): float(rating)
        for name, rating in zip(elo_df["team_name"], elo_df["rating"], strict=True)
        if pd.notna(name) and pd.notna(rating)
    }


def _elo_diff_for(home_team: str, away_team: str, elo_by_team: dict[str, float]) -> float:
    h = elo_by_team.get(home_team)
    a = elo_by_team.get(away_team)
    if h is None or a is None:
        return 0.0
    return float(h) - float(a)


def _resolve_competition_seasons(target_dir: Path) -> list[tuple[int, int]]:
    """Discover (competition_id, season_id) pairs from shots.parquet snapshots."""
    out: list[tuple[int, int]] = []
    for comp_dir in sorted(p for p in target_dir.iterdir() if p.is_dir()):
        if not comp_dir.name.isdigit():
            continue
        comp_id = int(comp_dir.name)
        for season_dir in sorted(p for p in comp_dir.iterdir() if p.is_dir()):
            if not season_dir.name.isdigit():
                continue
            if not (season_dir / "shots.parquet").exists():
                continue
            out.append((comp_id, int(season_dir.name)))
    return out


def _build_rows_from_one_match(
    match: dict[str, Any],
    *,
    elo_by_team: dict[str, float],
    session,
) -> list[dict[str, Any]]:
    home_team = (match.get("home_team") or {}).get("home_team_name")
    away_team = (match.get("away_team") or {}).get("away_team_name")
    home_score = match.get("home_score")
    away_score = match.get("away_score")
    if (
        home_team is None
        or away_team is None
        or home_score is None
        or away_score is None
    ):
        return []
    mid = int(match["match_id"])
    try:
        events = fetch_match_events(mid, session=session)
    except Exception:
        logger.warning("statsbomb fetch_match_events(%d) failed; skipping", mid)
        return []
    snapshots = replay_statsbomb_events(events, home_team=home_team, away_team=away_team)
    return snapshots_to_training_rows(
        snapshots,
        elo_diff=_elo_diff_for(home_team, away_team, elo_by_team),
        final_home_score=int(home_score),
        final_away_score=int(away_score),
    )


def build_training_rows(
    *,
    target_dir: Path = STATSBOMB_DEFAULT_TARGET,
    session=None,
    max_matches: int | None = None,
    elo_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Walk the StatsBomb tournaments on disk and emit a per-state-snapshot DataFrame.

    Returns an empty DataFrame (with the right columns) when no shots Parquet
    is on disk yet — that's the "fresh repo" case; callers should log and
    fall back to a synthetic training run.
    """
    pairs = _resolve_competition_seasons(target_dir)
    if not pairs:
        return pd.DataFrame(
            columns=["elo_diff", "goal_diff", "minutes_remaining", "red_diff", "label"]
        )
    if elo_df is None:
        try:
            elo_df = load_elo_snapshot()
        except FileNotFoundError:
            elo_df = None
    elo_by_team = _elo_by_team(elo_df)
    rows: list[dict[str, Any]] = []
    n_emitted = 0
    for comp_id, season_id in pairs:
        try:
            matches = fetch_matches(comp_id, season_id, session=session)
        except Exception:
            logger.warning("fetch_matches(%d, %d) failed; skipping", comp_id, season_id)
            continue
        for match in matches:
            new_rows = _build_rows_from_one_match(
                match, elo_by_team=elo_by_team, session=session
            )
            if not new_rows:
                continue
            rows.extend(new_rows)
            n_emitted += 1
            if max_matches is not None and n_emitted >= max_matches:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def fit_from_rows(rows: pd.DataFrame | Iterable[dict[str, Any]]) -> LiveWinProbModel:
    """Fit a ``LiveWinProbModel`` from a per-state-snapshot DataFrame or iterable."""
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    if df.empty:
        raise ValueError("no rows — populate the StatsBomb corpus first")
    feature_cols = ["elo_diff", "goal_diff", "minutes_remaining", "red_diff"]
    return LiveWinProbModel.fit(df[feature_cols], np.asarray(df["label"], dtype=int))


def fit_and_save(
    *,
    rows: pd.DataFrame | None = None,
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    max_matches: int | None = None,
) -> Path:
    """Top-level entrypoint. Returns the artefact path."""
    if rows is None:
        rows = build_training_rows(max_matches=max_matches)
    model = fit_from_rows(rows)
    out = model.save(artifact_path)
    logger.info("live win-prob: fit on %d rows; saved to %s", len(rows), out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Cap on the number of matches contributing rows (debug knob).",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    fit_and_save(max_matches=args.rows)


if __name__ == "__main__":
    main()


__all__ = [
    "build_training_rows",
    "fit_and_save",
    "fit_from_rows",
]
