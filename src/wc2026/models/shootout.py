"""Penalty-shootout submodel — Elo-difference logistic regression.

The base :func:`wc2026.sim.knockout.simulate_knockout_match` resolves a tied
extra-time match by a 50/50 coin flip. Empirically the shootout winner is
weakly but non-trivially predictable from Elo difference — stronger teams win
slightly more often, even controlling for the fact that they are more likely
to reach a shootout in the first place. Using the actual Elo gap as a logistic
feature is a small upgrade over 50/50 that also makes the platform less
embarrassing to inspect ("you really claim France and Andorra are coin-flips in
a shootout?").

Design choices
--------------
- **Single feature** ``elo_home - elo_away``: symmetry is enforced by setting
  ``fit_intercept=False``. At equal Elo the model predicts exactly 0.5.
- **scikit-learn LogisticRegression** for the fit (default L2, C=1.0).
  ``ShootoutModel`` is a thin wrapper that stores the fitted slope plus the
  Elo lookup dict.
- **Injectable, not invasive**: this module never imports from ``wc2026.sim``.
  The intended caller pattern (see :func:`simulate_shootout` docstring) is to
  post-process a ``KnockoutOutcome`` whose ``decided_in == "shootout"``,
  replacing the placeholder coin-flip winner. ``sim/knockout.py`` is not
  modified.

Reference
---------
World Football Elo Ratings, https://www.eloratings.net. Shootouts dataset
(``data/raw/jurisoo/shootouts.csv``) comes from the same Kaggle source as
``results.csv`` (martj42/international-football-results-from-1872-to-2017),
mirrored locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

DEFAULT_TARGET = Path("data/raw/jurisoo")


def load_historical_shootouts(target_dir: Path = DEFAULT_TARGET) -> pd.DataFrame:
    """Load ``shootouts.csv`` with typed columns and parsed dates.

    Returns a DataFrame with the columns:
        date        datetime64[ns]
        home_team   string
        away_team   string
        winner      string
    plus a derived bool ``home_team_won`` (True iff ``winner == home_team``).
    """
    path = target_dir / "shootouts.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Place the jurisoo shootouts.csv in {target_dir}."
        )
    df = pd.read_csv(path)
    expected = {"date", "home_team", "away_team", "winner"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"shootouts.csv missing columns: {sorted(missing)}")
    df = df.assign(
        date=pd.to_datetime(df["date"]),
        home_team=df["home_team"].astype("string"),
        away_team=df["away_team"].astype("string"),
        winner=df["winner"].astype("string"),
    )
    df["home_team_won"] = df["winner"] == df["home_team"]
    return df


def _elo_lookup(elo_snapshot: pd.DataFrame) -> dict[str, float]:
    """team_name -> rating (float). Last-write-wins on duplicates."""
    required = {"team_name", "rating"}
    missing = required - set(elo_snapshot.columns)
    if missing:
        raise ValueError(f"elo snapshot missing columns: {sorted(missing)}")
    return {
        str(t): float(r)
        for t, r in zip(elo_snapshot["team_name"], elo_snapshot["rating"], strict=True)
    }


@dataclass(frozen=True)
class ShootoutModel:
    """Fitted shootout submodel: P(home wins) = sigmoid(slope * (elo_h - elo_a))."""

    slope: float
    elo_lookup: dict[str, float]
    n_train: int

    def predict_home_win(self, home: str, away: str) -> float:
        """P(home team wins the shootout). 0.5 when either team has no Elo."""
        home_elo = self.elo_lookup.get(home)
        away_elo = self.elo_lookup.get(away)
        if home_elo is None or away_elo is None:
            return 0.5
        z = self.slope * (home_elo - away_elo)
        return float(1.0 / (1.0 + np.exp(-z)))


def fit_shootout_model(
    shootouts_df: pd.DataFrame,
    elo_snapshot_df: pd.DataFrame,
) -> ShootoutModel:
    """Fit a no-intercept logistic regression on Elo-difference.

    Rows whose home or away team is missing from ``elo_snapshot_df`` are
    dropped (they would otherwise drag the slope toward zero through the
    fallback). The fit only converges meaningfully if at least a handful of
    rows survive.
    """
    elo_lookup = _elo_lookup(elo_snapshot_df)
    if "home_team_won" not in shootouts_df.columns:
        df = shootouts_df.assign(home_team_won=shootouts_df["winner"] == shootouts_df["home_team"])
    else:
        df = shootouts_df
    mask = df["home_team"].isin(elo_lookup) & df["away_team"].isin(elo_lookup)
    used = df.loc[mask]
    if len(used) < 5:
        raise ValueError(
            f"only {len(used)} shootout rows have Elo for both teams; need >= 5 to fit"
        )
    x = (
        used["home_team"].map(elo_lookup).astype(float).to_numpy()
        - used["away_team"].map(elo_lookup).astype(float).to_numpy()
    ).reshape(-1, 1)
    y = used["home_team_won"].astype(int).to_numpy()
    clf = LogisticRegression(fit_intercept=False, C=1.0, solver="lbfgs")
    clf.fit(x, y)
    return ShootoutModel(slope=float(clf.coef_[0, 0]), elo_lookup=elo_lookup, n_train=len(used))


def predict_shootout(
    home: str,
    away: str,
    elo_snapshot: pd.DataFrame | None,
    model: ShootoutModel,
) -> float:
    """Return P(home wins). ``elo_snapshot`` is unused — the model captures the
    Elo lookup at fit time — but is kept in the signature so callers don't have
    to thread Elo through twice.
    """
    del elo_snapshot
    return model.predict_home_win(home, away)


def simulate_shootout(
    home: str,
    away: str,
    model: ShootoutModel,
    elo_snapshot: pd.DataFrame | None,
    rng: np.random.Generator,
) -> str:
    """Sample a shootout winner team name.

    Intended use: post-process the placeholder coin-flip winner from
    :func:`wc2026.sim.knockout.simulate_knockout_match` so we do not have
    to modify ``sim/knockout.py``. Example::

        outcome = simulate_knockout_match(home, away, dc_model, rng)
        if outcome.decided_in == "shootout":
            winner = simulate_shootout(home, away, shootout_model, None, rng)
            outcome = replace(outcome, winner=winner, shootout_winner=winner)
    """
    p = predict_shootout(home, away, elo_snapshot, model)
    return home if rng.random() < p else away
