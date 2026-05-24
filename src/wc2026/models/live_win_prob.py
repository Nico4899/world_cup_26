"""In-match live win-probability model.

Multinomial logistic regression on the four state features the Phase 6 plan
calls out:

    - ``elo_diff``         Pre-match home Elo minus away Elo.
    - ``goal_diff``        Current home goals minus away goals.
    - ``minutes_remaining`` 90 (regulation) or 120 (ET) minus the current minute.
    - ``red_diff``         Current home red-cards minus away red-cards.

Output is P(home_win | state), P(draw | state), P(away_win | state) using the
same H/D/A class encoding as ``models.xgb_classifier``.

The model is intentionally small (5 parameters per class with no interaction
terms) so it stays interpretable on the dashboard's live win-prob chart and
calibrates quickly on the StatsBomb open-data corpus (~250 matches x ~120
state snapshots = 30k training rows).

Persistence: a tiny JSON file holding the coefficient matrix + intercepts +
feature ordering. No external libraries beyond sklearn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from wc2026.models.xgb_classifier import CLASS_AWAY, CLASS_DRAW, CLASS_HOME, N_CLASSES

DEFAULT_FEATURE_COLUMNS: tuple[str, ...] = (
    "elo_diff",
    "goal_diff",
    "minutes_remaining",
    "red_diff",
)

DEFAULT_ARTIFACT_PATH = Path("data/artifacts/live_win_prob/latest.json")

REGULATION_LENGTH = 90
EXTRA_TIME_LENGTH = 120


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically-stable row-wise softmax."""
    z = logits - logits.max(axis=-1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)


@dataclass(frozen=True)
class LiveWinProbModel:
    """Serialisable softmax model for live in-match probabilities."""

    intercepts: tuple[float, float, float]
    coefficients: tuple[tuple[float, ...], ...]  # shape (3 classes, n_features)
    feature_names: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS
    version: str = "live_win_prob.v1"

    @classmethod
    def fit(
        cls,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        *,
        sample_weight: np.ndarray | None = None,
        feature_names: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
    ) -> LiveWinProbModel:
        """Fit a multinomial logistic regression on ``X``.

        ``y`` must be integer labels in {0=H, 1=D, 2=A}. ``X`` must contain
        ``feature_names`` as columns; extras are ignored.
        """
        missing = [c for c in feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"X is missing required columns: {missing}")
        X_arr = X[list(feature_names)].to_numpy(dtype=float)
        y_arr = np.asarray(y, dtype=int)
        if y_arr.ndim != 1 or len(y_arr) != len(X_arr):
            raise ValueError("y must be a 1-d array the same length as X")
        if not set(np.unique(y_arr)).issubset({CLASS_HOME, CLASS_DRAW, CLASS_AWAY}):
            raise ValueError("y labels must be a subset of {CLASS_HOME, CLASS_DRAW, CLASS_AWAY}")
        clf = LogisticRegression(C=1.0, max_iter=500)
        clf.fit(
            X_arr,
            y_arr,
            sample_weight=(
                None if sample_weight is None else np.asarray(sample_weight, dtype=float)
            ),
        )
        # sklearn's multinomial logistic stores classes in `clf.classes_` in
        # ascending order — re-order so row 0 is H, row 1 is D, row 2 is A
        # regardless of which classes were seen during fit.
        n_classes_present = len(clf.classes_)
        coef = np.zeros((N_CLASSES, X_arr.shape[1]))
        intercept = np.zeros(N_CLASSES)
        for raw_idx, label in enumerate(clf.classes_):
            class_row = clf.coef_[raw_idx] if n_classes_present > 1 else clf.coef_[0]
            class_intercept = (
                clf.intercept_[raw_idx] if n_classes_present > 1 else clf.intercept_[0]
            )
            coef[int(label)] = class_row
            intercept[int(label)] = class_intercept
        return cls(
            intercepts=tuple(float(v) for v in intercept),
            coefficients=tuple(tuple(float(c) for c in row) for row in coef),
            feature_names=tuple(feature_names),
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(home_win, draw, away_win) for every row in ``X``.

        Missing columns get NaN → 0 imputation so a clean lifespan-time
        bootstrap (no events ingested yet) doesn't 500 the API.
        """
        df = X.copy()
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0
        ordered = (
            df[list(self.feature_names)]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        coef = np.asarray(self.coefficients)
        intercept = np.asarray(self.intercepts)
        logits = ordered @ coef.T + intercept
        return _softmax(logits)

    def predict_one(
        self,
        *,
        elo_diff: float,
        goal_diff: int,
        minutes_remaining: int,
        red_diff: int,
    ) -> dict[str, float]:
        """Convenience: predict for one state, return ``{home_win, draw, away_win}``."""
        df = pd.DataFrame(
            [
                {
                    "elo_diff": float(elo_diff),
                    "goal_diff": int(goal_diff),
                    "minutes_remaining": int(minutes_remaining),
                    "red_diff": int(red_diff),
                }
            ]
        )
        probs = self.predict_proba(df)[0]
        return {
            "home_win": float(probs[CLASS_HOME]),
            "draw": float(probs[CLASS_DRAW]),
            "away_win": float(probs[CLASS_AWAY]),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "intercepts": list(self.intercepts),
            "coefficients": [list(row) for row in self.coefficients],
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LiveWinProbModel:
        return cls(
            intercepts=tuple(float(v) for v in payload["intercepts"]),
            coefficients=tuple(tuple(float(c) for c in row) for row in payload["coefficients"]),
            feature_names=tuple(payload.get("feature_names", DEFAULT_FEATURE_COLUMNS)),
            version=str(payload.get("version", "live_win_prob.v1")),
        )

    def save(self, path: Path = DEFAULT_ARTIFACT_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path = DEFAULT_ARTIFACT_PATH) -> LiveWinProbModel:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def minutes_remaining_from_minute(minute: int, period: int = 1) -> int:
    """Map ``(minute, period)`` to the documented ``minutes_remaining`` feature.

    Period 1/2: 90 - minute (regulation).
    Period 3/4: 120 - minute (extra time).
    Period 5: 0 (penalty shootout — model isn't designed for it; caller should
              fall back to the shootout submodel).
    """
    if period >= 5:
        return 0
    if period >= 3:
        return max(EXTRA_TIME_LENGTH - minute, 0)
    return max(REGULATION_LENGTH - minute, 0)


# Synonyms kept so callers don't have to import private names from xgb_classifier.
@dataclass(frozen=True)
class LiveStateRow:
    """One state snapshot for training or prediction (column-aligned with the model)."""

    elo_diff: float
    goal_diff: int
    minutes_remaining: int
    red_diff: int
    label: int | None = field(default=None)

    def as_features(self) -> dict[str, float]:
        return {
            "elo_diff": float(self.elo_diff),
            "goal_diff": float(self.goal_diff),
            "minutes_remaining": float(self.minutes_remaining),
            "red_diff": float(self.red_diff),
        }


__all__ = [
    "DEFAULT_ARTIFACT_PATH",
    "DEFAULT_FEATURE_COLUMNS",
    "EXTRA_TIME_LENGTH",
    "REGULATION_LENGTH",
    "LiveStateRow",
    "LiveWinProbModel",
    "minutes_remaining_from_minute",
]
