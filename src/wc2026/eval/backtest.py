"""Hindcast harness: replay a list of matches day-by-day, predicting each from
a model fit only on data strictly before that day.

This is the trustworthiness gate for the platform — see the plan file's
"single end-to-end test that determines whether the platform is honest".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from wc2026.eval.calibration import observed_outcome
from wc2026.features.match_weights import combined_weight
from wc2026.models.poisson_dc import PoissonDC

DEFAULT_HALF_LIFE_DAYS: float = 3650.0
"""Selected by the Stage 0.6 sweep on WC 2022 (log-loss minimum at ~10y; longer
values give only marginal improvement). See scripts/calibration_sweep.py."""

DEFAULT_HISTORY_WINDOW_YEARS: float = 10.0


@dataclass(frozen=True)
class HindcastConfig:
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS
    history_window_years: float = DEFAULT_HISTORY_WINDOW_YEARS
    refit_cadence: str = "daily"  # "daily" or "per_match"


def _train_for_cutoff(
    history: pd.DataFrame,
    cutoff: pd.Timestamp,
    cfg: HindcastConfig,
) -> tuple[PoissonDC, int]:
    """Build training set + weights for matches strictly before ``cutoff`` and fit."""
    window_start = cutoff - pd.Timedelta(days=cfg.history_window_years * 365.25)
    train = history[(history["date"] >= window_start) & (history["date"] < cutoff)]
    train = train.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
    if train.empty:
        raise ValueError(f"empty training set for cutoff {cutoff.date()}")
    weights = combined_weight(train, ref_date=cutoff, half_life_days=cfg.half_life_days)
    model = PoissonDC().fit(train, weights=weights)
    return model, len(train)


def hindcast(
    target_matches: pd.DataFrame,
    history: pd.DataFrame,
    *,
    cfg: HindcastConfig | None = None,
    progress: bool = False,
    progress_out: Path | None = None,
) -> pd.DataFrame:
    """Predict each target match using a model fit on strictly-earlier history.

    Parameters
    ----------
    target_matches :
        DataFrame of matches to predict. Required columns: ``date`` (datetime),
        ``home_team``, ``away_team``, ``home_score``, ``away_score``, ``neutral``.
    history :
        Full historical played-match DataFrame (typically ``load_played()``).
    cfg :
        Hindcast configuration (half-life, history window, refit cadence).
    progress :
        If True, print a one-line status per refit.

    Returns
    -------
    DataFrame with one row per target match and the columns:
        date, home_team, away_team, observed, actual_home, actual_away,
        p_home, p_draw, p_away, train_n, neutral, skipped_reason
    """
    cfg = cfg or HindcastConfig()
    required = {"date", "home_team", "away_team", "home_score", "away_score", "neutral"}
    missing = required - set(target_matches.columns)
    if missing:
        raise ValueError(f"target_matches missing columns: {sorted(missing)}")

    target = target_matches.sort_values("date").reset_index(drop=True)
    records: list[dict] = []
    cached_model: PoissonDC | None = None
    cached_train_n: int = 0
    cached_cutoff: pd.Timestamp | None = None

    for _, m in target.iterrows():
        cutoff = pd.Timestamp(m["date"])
        need_refit = cached_model is None or (
            cfg.refit_cadence == "daily"
            and (cached_cutoff is None or cutoff.date() != cached_cutoff.date())
        )
        if cfg.refit_cadence == "per_match":
            need_refit = True
        if need_refit:
            cached_model, cached_train_n = _train_for_cutoff(history, cutoff, cfg)
            cached_cutoff = cutoff
            if progress:
                msg = (
                    f"[hindcast] refit cutoff={cutoff.date()} "
                    f"train_n={cached_train_n} converged={cached_model.converged_} "
                    f"home_adv={cached_model.params_.home_advantage:.3f} "
                    f"rho={cached_model.params_.rho:.4f}"
                )
                print(msg)
                if progress_out:
                    progress_out.parent.mkdir(parents=True, exist_ok=True)
                    with progress_out.open("a") as f:
                        f.write(msg + "\n")

        actual_h = int(m["home_score"]) if pd.notna(m["home_score"]) else None
        actual_a = int(m["away_score"]) if pd.notna(m["away_score"]) else None
        observed = observed_outcome(actual_h, actual_a) if actual_h is not None else None

        try:
            probs = cached_model.outcome_probs(
                m["home_team"], m["away_team"], neutral=bool(m["neutral"])
            )
            p_h, p_d, p_a = probs["home_win"], probs["draw"], probs["away_win"]
            skipped = None
        except KeyError as e:
            p_h = p_d = p_a = float("nan")
            skipped = str(e)

        records.append(
            {
                "date": m["date"],
                "home_team": m["home_team"],
                "away_team": m["away_team"],
                "neutral": bool(m["neutral"]),
                "observed": observed,
                "actual_home": actual_h,
                "actual_away": actual_a,
                "p_home": p_h,
                "p_draw": p_d,
                "p_away": p_a,
                "train_n": cached_train_n,
                "skipped_reason": skipped,
            }
        )

    return pd.DataFrame(records)
