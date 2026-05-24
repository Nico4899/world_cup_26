"""Backtest PoissonDC against Bet365/Pinnacle closing odds on a club-football
corpus.

Why this exists
---------------
The published bookmaker reference values on `/track-record` (e.g. WC 2018
log-loss ≈ 0.96-1.00) come from peer-reviewed sources (Wheatcroft 2019;
Constantinou 2019), not from a corpus we hold. The football-data.co.uk
data we ingest covers domestic leagues only — there is no public closing-
odds aggregate for the World Cup itself. To check that *our model
architecture* is bookmaker-competitive at all, we hindcast PoissonDC on
the same club-football corpus and compare to its closing odds.

Output
------
Prints a single block to stdout with the comparison metrics; writes a
JSON summary to ``data/artifacts/bookmaker_benchmark/latest.json``. The
methodology doc links to this script so reviewers can reproduce.

Usage
-----
    uv run python scripts/backtest_against_bookmaker.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from wc2026.eval.calibration import match_log_loss
from wc2026.features.match_weights import combined_weight
from wc2026.ingest.football_data_co_uk import (
    DEFAULT_CALIBRATION_SET,
    DEFAULT_TARGET,
    implied_probabilities,
    parse_csv,
)
from wc2026.models.poisson_dc import PoissonDC

DEFAULT_HOLDOUT_FROM = pd.Timestamp("2024-08-01")
DEFAULT_HALF_LIFE_DAYS = 365.0  # Club football: shorter half-life than international.
DEFAULT_OUTPUT = Path("data/artifacts/bookmaker_benchmark/latest.json")

logger = logging.getLogger(__name__)


def _load_corpus(
    target_dir: Path,
    seasons: tuple[tuple[str, str], ...] = DEFAULT_CALIBRATION_SET,
) -> pd.DataFrame:
    """Concatenate every cached league CSV into one DataFrame."""
    frames: list[pd.DataFrame] = []
    for season_code, league_code in seasons:
        path = target_dir / f"{league_code}_{season_code}.csv"
        if not path.exists():
            logger.warning("missing csv at %s — skipping", path)
            continue
        df = parse_csv(path.read_text(encoding="latin-1"))
        if df.empty:
            continue
        df = df[df["ftr"].isin({"H", "D", "A"})]
        df["league_code"] = league_code
        df["season_code"] = season_code
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No football-data.co.uk CSVs found under {target_dir}. "
            "Run scripts/scrape_football_data_co_uk.py first."
        )
    return pd.concat(frames, ignore_index=True).sort_values("match_date").reset_index(drop=True)


def _fit_poisson(train: pd.DataFrame, ref_date: pd.Timestamp) -> PoissonDC:
    """Fit PoissonDC on the club-football training rows.

    The corpus has no match-importance gradations (it's all league play),
    so the weighting is pure exponential time-decay with a 1-year half-life
    — shorter than the international model (3650 days) because club squads
    turn over much faster.
    """
    df = train.rename(columns={"fthg": "home_score", "ftag": "away_score"}).copy()
    df["neutral"] = False
    df["tournament"] = "Domestic league"
    # combined_weight expects a `date` column (pd.Timestamp).
    df["date"] = pd.to_datetime(df["match_date"])
    df = df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
    weights = combined_weight(df, ref_date=ref_date, half_life_days=DEFAULT_HALF_LIFE_DAYS)
    return PoissonDC().fit(df, weights=weights)


def _predict_and_score(
    model: PoissonDC, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-match (poisson_logloss, bookmaker_logloss, observed_class).

    Skips rows where either side is missing from the training set or the
    bookmaker odds are unavailable.
    """
    with_odds = implied_probabilities(test).dropna(
        subset=["p_home", "p_draw", "p_away"]
    )
    poisson_ll: list[float] = []
    bookmaker_ll: list[float] = []
    observed: list[str] = []
    for _, row in with_odds.iterrows():
        try:
            probs = model.outcome_probs(row["home_team"], row["away_team"], neutral=False)
        except KeyError:
            continue
        obs = row["ftr"]
        if obs not in {"H", "D", "A"}:
            continue
        poisson_ll.append(
            match_log_loss(obs, probs["home_win"], probs["draw"], probs["away_win"])
        )
        bookmaker_ll.append(
            match_log_loss(obs, row["p_home"], row["p_draw"], row["p_away"])
        )
        observed.append(obs)
    return np.array(poisson_ll), np.array(bookmaker_ll), np.array(observed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(DEFAULT_TARGET))
    parser.add_argument(
        "--holdout-from",
        default=str(DEFAULT_HOLDOUT_FROM.date()),
        help="Inclusive cutoff date — matches on or after this date are held out.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    corpus = _load_corpus(Path(args.corpus))
    cutoff = pd.Timestamp(args.holdout_from)
    train = corpus[corpus["match_date"] < cutoff].reset_index(drop=True)
    test = corpus[corpus["match_date"] >= cutoff].reset_index(drop=True)
    if train.empty or test.empty:
        logger.error("empty train (%d) or test (%d) split", len(train), len(test))
        return 1

    logger.info(
        "fitting PoissonDC on %d club matches (cutoff=%s); %d held-out matches",
        len(train),
        cutoff.date(),
        len(test),
    )
    model = _fit_poisson(train, ref_date=cutoff)
    poisson_ll, bookmaker_ll, obs = _predict_and_score(model, test)

    summary = {
        "as_of": datetime.now(UTC).isoformat(),
        "cutoff": str(cutoff.date()),
        "n_train": len(train),
        "n_test": len(test),
        "n_scored": len(poisson_ll),
        "poisson_log_loss": float(poisson_ll.mean()) if len(poisson_ll) else None,
        "bookmaker_log_loss": float(bookmaker_ll.mean()) if len(bookmaker_ll) else None,
        "delta": float(poisson_ll.mean() - bookmaker_ll.mean()) if len(poisson_ll) else None,
        "base_h": float((obs == "H").mean()) if len(obs) else None,
        "base_d": float((obs == "D").mean()) if len(obs) else None,
        "base_a": float((obs == "A").mean()) if len(obs) else None,
        "leagues": sorted(
            {
                (season, league)
                for season, league in zip(
                    corpus["season_code"], corpus["league_code"], strict=True
                )
            }
        ),
        "half_life_days": DEFAULT_HALF_LIFE_DAYS,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print()
    print("  Club-football PoissonDC vs Bet365/Pinnacle closing odds")
    print("  ------------------------------------------------------")
    print(f"  Train matches  : {summary['n_train']:>5d}")
    print(f"  Test matches   : {summary['n_test']:>5d}")
    print(
        f"  Scored matches : {summary['n_scored']:>5d}  "
        f"(skipped rows lacked odds or trained teams)"
    )
    if summary["n_scored"]:
        print(f"  PoissonDC log-loss   : {summary['poisson_log_loss']:.4f}")
        print(f"  Bookmaker log-loss   : {summary['bookmaker_log_loss']:.4f}")
        print(f"  Delta (model - book) : {summary['delta']:+.4f}")
    print(f"  Written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
