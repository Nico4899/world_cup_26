"""WC 2018 hindcast — same logic as backtest_wc2022.py but for the 2018 tournament.

Used to confirm that the half-life tuned on WC 2022 generalises to a different
tournament. If WC 2018 log-loss is materially worse than WC 2022, we've overfit
to one tournament; expect them to be similar.
"""

from __future__ import annotations

import time

import pandas as pd

from wc2026.eval.backtest import HindcastConfig, hindcast
from wc2026.eval.calibration import aggregate, base_rates, baseline_log_loss
from wc2026.ingest.kaggle_intl import load_played

WC2018_START = pd.Timestamp("2018-06-14")
WC2018_END = pd.Timestamp("2018-07-15")


def main() -> int:
    history = load_played()
    target = history[
        (history["tournament"] == "FIFA World Cup")
        & (history["date"] >= WC2018_START)
        & (history["date"] <= WC2018_END)
    ].copy()
    print(f"predicting {len(target)} WC 2018 matches")

    t0 = time.time()
    preds = hindcast(target, history, cfg=HindcastConfig())  # defaults
    dt = time.time() - t0
    print(f"hindcast complete in {dt:.1f}s ({dt / len(preds) * 1000:.0f}ms/match)")

    clean = preds.dropna(subset=["p_home", "p_draw", "p_away", "observed"])
    metrics = aggregate(clean)
    print()
    print("=== overall metrics (lower = better) ===")
    print(f"  log-loss: {metrics.log_loss:.4f}")
    print(f"  Brier:    {metrics.brier:.4f}")
    print(f"  RPS:      {metrics.rps:.4f}")

    obs = clean["observed"].tolist()
    rates = base_rates(obs)
    print()
    print("=== references ===")
    print(
        "  WC2018 base rates: H={:.2%}  D={:.2%}  A={:.2%}".format(
            rates["H"], rates["D"], rates["A"]
        )
    )
    print(f"  climatological log-loss:        {baseline_log_loss(obs):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
