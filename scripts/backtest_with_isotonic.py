"""WC 2022 hindcast with leave-one-out isotonic recalibration.

Usage:
    uv run python scripts/backtest_with_isotonic.py

Runs the standard WC 2022 hindcast, then applies LOO per-outcome isotonic
recalibration. Prints log-loss / Brier / RPS before and after, plus the
per-outcome reliability diagram pre vs post so you can see whether the
calibrator is actually moving things in the right direction.
"""

from __future__ import annotations

import time

import pandas as pd

from wc2026.eval.backtest import HindcastConfig, hindcast
from wc2026.eval.calibration import aggregate, reliability_diagram
from wc2026.eval.isotonic import leave_one_out_recalibrate
from wc2026.ingest.kaggle_intl import load_played

WC2022_START = pd.Timestamp("2022-11-20")
WC2022_END = pd.Timestamp("2022-12-18")


def _print_reliability(predictions: pd.DataFrame, label: str) -> None:
    print(f"=== reliability diagram ({label}) ===")
    print("  outcome  bin range      n   mean_pred  realized")
    for outcome in ("H", "D", "A"):
        for b in reliability_diagram(predictions, n_bins=10):
            if b.outcome != outcome or b.n == 0:
                continue
            print(
                f"  {b.outcome:>7s}  [{b.bin_low:.1f},{b.bin_high:.1f})  "
                f"{b.n:>4}  {b.mean_predicted:>9.3f}  {b.realized_frequency:>8.3f}"
            )
    print()


def main() -> int:
    history = load_played()
    target = history[
        (history["tournament"] == "FIFA World Cup")
        & (history["date"] >= WC2022_START)
        & (history["date"] <= WC2022_END)
    ].copy()
    print(f"WC 2022 hindcast + LOO isotonic recalibration ({len(target)} matches)")

    t0 = time.time()
    preds = hindcast(target, history, cfg=HindcastConfig())
    raw = preds.dropna(subset=["p_home", "p_draw", "p_away", "observed"]).copy()
    print(f"hindcast done in {time.time() - t0:.1f}s; {len(raw)} clean predictions")

    cal = leave_one_out_recalibrate(raw)

    m_raw = aggregate(raw)
    m_cal = aggregate(cal)

    print()
    print(f"{'variant':>18s} {'n':>5s} {'log_loss':>10s} {'brier':>8s} {'rps':>8s}")
    print("-" * 52)
    print(
        f"{'raw PoissonDC':>18s} {m_raw.n:>5d} {m_raw.log_loss:>10.4f} {m_raw.brier:>8.4f} {m_raw.rps:>8.4f}"
    )
    print(
        f"{'+ LOO isotonic':>18s} {m_cal.n:>5d} {m_cal.log_loss:>10.4f} {m_cal.brier:>8.4f} {m_cal.rps:>8.4f}"
    )
    delta_ll = m_cal.log_loss - m_raw.log_loss
    print()
    print(f"delta log_loss: {delta_ll:+.4f}  ({'improvement' if delta_ll < 0 else 'degradation'})")
    print()

    _print_reliability(raw, "raw")
    _print_reliability(cal, "after LOO isotonic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
