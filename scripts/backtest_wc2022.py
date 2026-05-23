"""WC 2022 day-by-day hindcast.

Usage:
    uv run python scripts/backtest_wc2022.py

For each of the 64 WC 2022 matches, fit the bivariate-Poisson + Dixon-Coles
model on all played international matches strictly before the match date,
and predict the 1X2 outcome. Aggregate Brier/log-loss/RPS and report.
"""

from __future__ import annotations

import time

import pandas as pd

from wc2026.eval.backtest import HindcastConfig, hindcast
from wc2026.eval.calibration import (
    aggregate,
    base_rates,
    baseline_log_loss,
    reliability_diagram,
)
from wc2026.ingest.kaggle_intl import load_played

WC2022_START = pd.Timestamp("2022-11-20")
WC2022_END = pd.Timestamp("2022-12-18")


def main() -> int:
    history = load_played()
    target = history[
        (history["tournament"] == "FIFA World Cup")
        & (history["date"] >= WC2022_START)
        & (history["date"] <= WC2022_END)
    ].copy()
    print(f"predicting {len(target)} WC 2022 matches")

    t0 = time.time()
    preds = hindcast(target, history, cfg=HindcastConfig())  # defaults
    dt = time.time() - t0
    print(f"hindcast complete in {dt:.1f}s ({dt / len(preds) * 1000:.0f}ms/match)")
    n_skipped = int(preds["skipped_reason"].notna().sum())
    print(f"skipped: {n_skipped}")

    clean = preds.dropna(subset=["p_home", "p_draw", "p_away", "observed"]).copy()
    print(f"evaluating {len(clean)} matches with full predictions")

    metrics = aggregate(clean)
    print()
    print("=== overall metrics (lower = better) ===")
    print(f"  log-loss: {metrics.log_loss:.4f}")
    print(f"  Brier:    {metrics.brier:.4f}")
    print(f"  RPS:      {metrics.rps:.4f}")
    print()

    obs = clean["observed"].tolist()
    rates = base_rates(obs)
    print("=== references ===")
    print("  uniform (1/3,1/3,1/3) log-loss = 1.0986")
    print(
        "  WC2022 base rates: H={:.2%}  D={:.2%}  A={:.2%}".format(
            rates["H"], rates["D"], rates["A"]
        )
    )
    print(f"  climatological log-loss:        {baseline_log_loss(obs):.4f}")
    print()

    for outcome, label in [("H", "Home"), ("D", "Draw"), ("A", "Away")]:
        print(f"=== reliability diagram ({label} outcome) ===")
        print("  bin range      n   mean_pred  realized")
        for b in reliability_diagram(clean, n_bins=10):
            if b.outcome != outcome or b.n == 0:
                continue
            print(
                f"  [{b.bin_low:.1f},{b.bin_high:.1f})  {b.n:>4}  "
                f"{b.mean_predicted:>9.3f}  {b.realized_frequency:>8.3f}"
            )
        print()

    # Most surprising predictions: realized outcome had lowest probability
    clean["surprise"] = clean.apply(
        lambda r: 1 - r[f"p_{ {'H': 'home', 'D': 'draw', 'A': 'away'}[r['observed']] }"],
        axis=1,
    )
    print("=== top 5 most surprising results (realized outcome had lowest forecast) ===")
    top = clean.sort_values("surprise", ascending=False).head(5)
    for _, r in top.iterrows():
        print(
            f"  {r['date'].date()} {r['home_team']} {r['actual_home']}-{r['actual_away']} {r['away_team']}: "
            f"p_H={r['p_home']:.2f}, p_D={r['p_draw']:.2f}, p_A={r['p_away']:.2f}  (observed {r['observed']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
