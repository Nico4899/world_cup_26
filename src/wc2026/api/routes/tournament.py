"""Tournament-level routes: aggregated Monte Carlo standings + sample bracket."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, Query

from wc2026.api.dependencies import get_fixtures, get_model
from wc2026.models.poisson_dc import PoissonDC
from wc2026.sim.fixtures import WC2026Fixtures
from wc2026.sim.tournament import (
    ROUND_COLUMNS,
    simulate_tournament,
    simulate_tournament_monte_carlo,
)

router = APIRouter(prefix="/api/v1/tournament")

DEFAULT_N_SIMS = 2000
DEFAULT_SEED = 42
CACHE_TTL_SECONDS = 3600  # 1 hour

# Module-level cache keyed by (n_sims, seed). Thread-safe because GIL + lock around mutation.
_CACHE: dict[tuple[int, int], tuple[float, Any]] = {}
_CACHE_LOCK = threading.Lock()


def _cached_summary(model: PoissonDC, fixtures: WC2026Fixtures, n_sims: int, seed: int) -> Any:
    key = (n_sims, seed)
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None and now - cached[0] < CACHE_TTL_SECONDS:
            return cached[1]
    summary = simulate_tournament_monte_carlo(fixtures, model, n_sims=n_sims, seed=seed)
    with _CACHE_LOCK:
        _CACHE[key] = (now, summary)
    return summary


def _build_group_block(letter: str, summary, fixtures: WC2026Fixtures) -> dict[str, Any]:
    teams = list(fixtures.groups[letter])
    rows = []
    for team in teams:
        row = summary.probabilities.loc[team]
        rows.append(
            {
                "team": team,
                "p_first": float(row["group_winner"]),
                "p_second": float(row["runner_up"]),
                "p_third_advance": float(row["third_advance"]),
                # P(eliminated in group) = 1 - (sum of three advance routes)
                "p_eliminated": float(
                    max(
                        0.0,
                        1.0 - row["group_winner"] - row["runner_up"] - row["third_advance"],
                    )
                ),
            }
        )
    rows.sort(key=lambda r: r["p_first"] + r["p_second"] + r["p_third_advance"], reverse=True)
    return {"group": letter, "teams": rows}


@router.get("/standings")
def standings(
    n_sims: int = Query(default=DEFAULT_N_SIMS, ge=100, le=20_000),
    seed: int = Query(default=DEFAULT_SEED),
    model: PoissonDC = Depends(get_model),
    fixtures: WC2026Fixtures = Depends(get_fixtures),
) -> dict[str, Any]:
    summary = _cached_summary(model, fixtures, n_sims, seed)
    groups = [_build_group_block(letter, summary, fixtures) for letter in fixtures.groups]
    # Top-10 championship probabilities for the headline
    top = (
        summary.probabilities.sort_values("champion", ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"index": "team"})
    )
    headline = [
        {
            "team": r["team"],
            "p_champion": float(r["champion"]),
            "p_final": float(r["final_reached"]),
            "p_sf": float(r["sf_reached"]),
            "p_qf": float(r["qf_reached"]),
        }
        for _, r in top.iterrows()
    ]
    return {
        "n_sims": int(summary.n_sims),
        "round_columns": list(ROUND_COLUMNS),
        "groups": groups,
        "headline": headline,
    }


@router.get("/bracket")
def bracket(
    seed: int = Query(default=DEFAULT_SEED, description="Sample one bracket realisation."),
    model: PoissonDC = Depends(get_model),
    fixtures: WC2026Fixtures = Depends(get_fixtures),
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    result = simulate_tournament(fixtures, model, rng)
    matches = []
    for mid, outcome in sorted(result.knockout_results.items()):
        # round_label by match id range per FIFA numbering
        if 73 <= mid <= 88:
            round_label = "R32"
        elif 89 <= mid <= 96:
            round_label = "R16"
        elif 97 <= mid <= 100:
            round_label = "QF"
        elif 101 <= mid <= 102:
            round_label = "SF"
        else:
            round_label = "Final"
        matches.append(
            {
                "match_id": mid,
                "round": round_label,
                "home_team": outcome.home_team,
                "away_team": outcome.away_team,
                "winner": outcome.winner,
                "decided_in": outcome.decided_in,
                "regulation_score": list(outcome.regulation_score),
            }
        )
    return {
        "seed": seed,
        "champion": result.champion,
        "matches": matches,
    }
