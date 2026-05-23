"""Sweep the half-life hyperparameter on the WC 2022 hindcast.

The 730-day (2-year) default is a guess; the plan calls for picking the value
that minimises out-of-sample log-loss on WC 2022. Range 365-2190 days
(1-6 years).
"""

from __future__ import annotations

import time

import pandas as pd

from wc2026.eval.backtest import HindcastConfig, hindcast
from wc2026.eval.calibration import aggregate
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

    half_lives = [365, 547, 730, 1095, 1460, 1825, 2190, 2920, 3650, 5475, 36500]
    print(f"sweeping half_life across {half_lives} on {len(target)} WC 2022 matches")
    print()
    print(f"{'half_life':>10}  {'log_loss':>9}  {'Brier':>9}  {'RPS':>9}  {'time':>6}")
    print("-" * 55)
    rows = []
    best = (float("inf"), None)
    for hl in half_lives:
        t0 = time.time()
        preds = hindcast(target, history, cfg=HindcastConfig(half_life_days=hl, history_window_years=10))
        clean = preds.dropna(subset=["p_home", "p_draw", "p_away", "observed"])
        m = aggregate(clean)
        dt = time.time() - t0
        print(f"{hl:>10}  {m.log_loss:>9.4f}  {m.brier:>9.4f}  {m.rps:>9.4f}  {dt:>5.1f}s")
        rows.append((hl, m.log_loss, m.brier, m.rps))
        if m.log_loss < best[0]:
            best = (m.log_loss, hl)
    print("-" * 55)
    print(f"best half_life: {best[1]} days, log_loss = {best[0]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
