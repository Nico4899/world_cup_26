"""Calibration metrics for 1X2 predictions: log-loss, Brier, RPS, reliability.

Definitions match Gneiting & Raftery (2007, "Strictly Proper Scoring Rules,
Prediction, and Estimation", JASA 102:359-378). For an ordered three-outcome
forecast (home win / draw / away win) and an observed outcome, with predicted
probabilities ``(p_H, p_D, p_A)``:

  log_loss   = -log(p_{observed})
  brier      = sum_o (p_o - I{observed == o})^2
  rps        = 1/(K-1) * sum_{k=1..K-1} (cum_p_k - cum_r_k)^2
               with K = 3 outcomes ordered (H, D, A)

All three are negatively oriented (lower = better).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

OUTCOMES: tuple[str, str, str] = ("H", "D", "A")
EPS_LOG: float = 1e-15


def observed_outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def match_log_loss(observed: str, p_home: float, p_draw: float, p_away: float) -> float:
    p = {"H": p_home, "D": p_draw, "A": p_away}[observed]
    return -math.log(max(p, EPS_LOG))


def match_brier(observed: str, p_home: float, p_draw: float, p_away: float) -> float:
    realized = {"H": 0.0, "D": 0.0, "A": 0.0}
    realized[observed] = 1.0
    probs = {"H": p_home, "D": p_draw, "A": p_away}
    return sum((probs[o] - realized[o]) ** 2 for o in OUTCOMES)


def match_rps(observed: str, p_home: float, p_draw: float, p_away: float) -> float:
    """Ranked Probability Score with outcomes ordered (H, D, A)."""
    realized = {"H": 0.0, "D": 0.0, "A": 0.0}
    realized[observed] = 1.0
    # cumulative: after H, after H+D (the third cum is 1 == 1 so doesn't contribute)
    cum_p_1 = p_home
    cum_p_2 = p_home + p_draw
    cum_r_1 = realized["H"]
    cum_r_2 = realized["H"] + realized["D"]
    return 0.5 * ((cum_p_1 - cum_r_1) ** 2 + (cum_p_2 - cum_r_2) ** 2)


@dataclass(frozen=True)
class CalibrationMetrics:
    n: int
    log_loss: float
    brier: float
    rps: float

    def as_dict(self) -> dict[str, float]:
        return {"n": float(self.n), "log_loss": self.log_loss, "brier": self.brier, "rps": self.rps}


def aggregate(predictions: pd.DataFrame) -> CalibrationMetrics:
    """Compute mean log-loss / Brier / RPS over a DataFrame of predictions.

    Required columns: ``observed`` (H/D/A), ``p_home``, ``p_draw``, ``p_away``.
    """
    required = {"observed", "p_home", "p_draw", "p_away"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    if predictions.empty:
        return CalibrationMetrics(n=0, log_loss=float("nan"), brier=float("nan"), rps=float("nan"))
    lls = predictions.apply(
        lambda r: match_log_loss(r["observed"], r["p_home"], r["p_draw"], r["p_away"]), axis=1
    )
    bs = predictions.apply(
        lambda r: match_brier(r["observed"], r["p_home"], r["p_draw"], r["p_away"]), axis=1
    )
    rs = predictions.apply(
        lambda r: match_rps(r["observed"], r["p_home"], r["p_draw"], r["p_away"]), axis=1
    )
    return CalibrationMetrics(
        n=len(predictions),
        log_loss=float(lls.mean()),
        brier=float(bs.mean()),
        rps=float(rs.mean()),
    )


@dataclass(frozen=True)
class ReliabilityBin:
    outcome: str  # "H", "D", or "A"
    bin_low: float
    bin_high: float
    n: int
    mean_predicted: float
    realized_frequency: float


def reliability_diagram(predictions: pd.DataFrame, *, n_bins: int = 10) -> list[ReliabilityBin]:
    """Per-outcome reliability table.

    For each outcome o in {H, D, A}, partition the predicted probabilities
    ``p_o`` into ``n_bins`` equal-width bins on [0, 1]; within each bin report
    (n, mean predicted prob, realized frequency of o being the actual outcome).
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[ReliabilityBin] = []
    for outcome in OUTCOMES:
        col = f"p_{outcome.lower()}" if outcome != "D" else "p_draw"
        col = {"H": "p_home", "D": "p_draw", "A": "p_away"}[outcome]
        probs = predictions[col].to_numpy()
        realized = (predictions["observed"] == outcome).to_numpy().astype(float)
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            # include hi only in the last bin so [0,0.1), [0.1,0.2), ..., [0.9,1.0]
            if i == n_bins - 1:
                mask = (probs >= lo) & (probs <= hi)
            else:
                mask = (probs >= lo) & (probs < hi)
            n = int(mask.sum())
            if n == 0:
                out.append(
                    ReliabilityBin(
                        outcome=outcome,
                        bin_low=lo,
                        bin_high=hi,
                        n=0,
                        mean_predicted=float("nan"),
                        realized_frequency=float("nan"),
                    )
                )
                continue
            out.append(
                ReliabilityBin(
                    outcome=outcome,
                    bin_low=lo,
                    bin_high=hi,
                    n=n,
                    mean_predicted=float(probs[mask].mean()),
                    realized_frequency=float(realized[mask].mean()),
                )
            )
    return out


def baseline_log_loss(observed_outcomes: Iterable[str]) -> float:
    """Marginal (climatological) log-loss using the empirical H/D/A base rates.

    The marginal-rate model is a useful 'no-skill' floor: any predictive model
    should beat this. For international football the base rate is typically
    ~46% H, ~24% D, ~30% A.
    """
    obs = list(observed_outcomes)
    if not obs:
        return float("nan")
    n = len(obs)
    rates = {o: obs.count(o) / n for o in OUTCOMES}
    return -sum(rates[o] * math.log(max(rates[o], EPS_LOG)) for o in OUTCOMES)


def base_rates(observed_outcomes: Sequence[str]) -> dict[str, float]:
    n = len(observed_outcomes)
    if n == 0:
        return {o: float("nan") for o in OUTCOMES}
    return {o: observed_outcomes.count(o) / n for o in OUTCOMES}
