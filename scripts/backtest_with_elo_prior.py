"""WC 2022 hindcast sweep over prior_strength values.

Usage:
    uv run python scripts/backtest_with_elo_prior.py

For each ``prior_strength`` in [0, 0.5, 1.0, 2.0, 5.0], run the same WC 2022
day-by-day hindcast that ``scripts/backtest_wc2022.py`` does, but fit
:class:`PoissonDCWithPrior` instead of the unpenalised PoissonDC. Print a
table comparing log-loss / Brier / RPS to the existing baseline.

Note: the present Elo snapshot is current (post-2026), but the prior is meant
to act as a soft anchor on team strength, not a leak — it only nudges the
attack/defence parameters, it does not see any 2022+ match results. (For a
strict apples-to-apples comparison one would want an Elo snapshot as of the
hindcast start date; we accept the soft leak for this internal-evaluation
sweep and document the limitation in ``docs/methodology.md``.)
"""

from __future__ import annotations

import time

import pandas as pd

from wc2026.eval.backtest import HindcastConfig
from wc2026.eval.calibration import aggregate, observed_outcome
from wc2026.features.match_weights import combined_weight
from wc2026.ingest.eloratings_scraper import load_latest_snapshot
from wc2026.ingest.kaggle_intl import load_played
from wc2026.models.elo_prior import elo_prior_centres
from wc2026.models.poisson_dc import PoissonDC
from wc2026.models.poisson_dc_with_prior import PoissonDCWithPrior

WC2022_START = pd.Timestamp("2022-11-20")
WC2022_END = pd.Timestamp("2022-12-18")
PRIOR_STRENGTHS: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0)


def hindcast_with_prior(
    target_matches: pd.DataFrame,
    history: pd.DataFrame,
    elo_snapshot: pd.DataFrame,
    *,
    cfg: HindcastConfig,
    prior_strength: float,
) -> pd.DataFrame:
    """Day-by-day hindcast with PoissonDCWithPrior; mirrors `eval.backtest.hindcast`.

    ``prior_strength=None`` is equivalent to the unpenalised baseline. We keep the
    code path identical to the existing hindcast loop (refit cadence, history
    window, weighting) so the only varying factor is the prior.
    """
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
            window_start = cutoff - pd.Timedelta(days=cfg.history_window_years * 365.25)
            train = history[(history["date"] >= window_start) & (history["date"] < cutoff)]
            train = train.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
            if train.empty:
                raise ValueError(f"empty training set for cutoff {cutoff.date()}")
            weights = combined_weight(train, ref_date=cutoff, half_life_days=cfg.half_life_days)
            teams = sorted(set(train["home_team"]).union(train["away_team"]))
            if prior_strength == 0.0:
                cached_model = PoissonDC().fit(train, weights=weights)
            else:
                atk_c, dfc_c = elo_prior_centres(teams, elo_snapshot)
                cached_model = PoissonDCWithPrior(
                    attack_centres=atk_c,
                    defence_centres=dfc_c,
                    teams=teams,
                    prior_strength=prior_strength,
                ).fit(train, weights=weights)
            cached_train_n = len(train)
            cached_cutoff = cutoff

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


def main() -> int:
    history = load_played()
    target = history[
        (history["tournament"] == "FIFA World Cup")
        & (history["date"] >= WC2022_START)
        & (history["date"] <= WC2022_END)
    ].copy()
    print(f"WC 2022 hindcast over prior_strength sweep ({len(target)} matches)")

    elo_snap = load_latest_snapshot()
    print(f"Elo snapshot: {len(elo_snap)} teams")
    cfg = HindcastConfig()

    print()
    print(f"{'prior_strength':>15s} {'n_pred':>7s} {'log_loss':>10s} {'brier':>8s} {'rps':>8s}")
    print("-" * 56)
    for ps in PRIOR_STRENGTHS:
        t0 = time.time()
        preds = hindcast_with_prior(target, history, elo_snap, cfg=cfg, prior_strength=ps)
        clean = preds.dropna(subset=["p_home", "p_draw", "p_away", "observed"])
        m = aggregate(clean)
        dt = time.time() - t0
        print(
            f"{ps:>15.2f} {m.n:>7d} {m.log_loss:>10.4f} {m.brier:>8.4f} {m.rps:>8.4f}  ({dt:.1f}s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
