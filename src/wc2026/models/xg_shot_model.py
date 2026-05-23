"""Tiny logistic-regression xG model.

Why
---
The blueprint calls for a per-shot xG model trained on the StatsBomb open-data
corpus. The recipe is the standard one (Caley 2015; Decroos et al. 2018):
logistic regression of ``is_goal`` on shot geometry (distance, angle) plus
categorical features (body part, pattern of play). The fit is fast (≪1 s on
20k shots), the model is interpretable (coefficients have direct meaning),
and the result is competitive with the published StatsBomb xG numbers for
our coarse use case (per-match xG sums, not per-shot ranking).

Penalties are kept in the corpus and given a pattern_of_play="Penalty"
indicator — that's enough for the model to learn the ~0.79 goal rate without
a separate penalty submodel.

Coefficients persist as JSON under ``data/artifacts/xg_shot/latest.json``;
the API and feature-engineering layer load them on startup.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

DEFAULT_BODY_PART_LEVELS: tuple[str, ...] = (
    "Right Foot",
    "Left Foot",
    "Head",
    "Other",
)
DEFAULT_PATTERN_LEVELS: tuple[str, ...] = (
    "Open Play",
    "Free Kick",
    "Penalty",
    "Corner",
    "Kick Off",
)
# Reference levels (first in each tuple) are dropped to avoid collinearity.
REFERENCE_BODY_PART = DEFAULT_BODY_PART_LEVELS[0]
REFERENCE_PATTERN = DEFAULT_PATTERN_LEVELS[0]

DEFAULT_ARTIFACT_PATH = Path("data/artifacts/xg_shot/latest.json")


@dataclass(frozen=True)
class XgShotModel:
    """Serialisable logistic-regression xG model."""

    intercept: float
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    body_part_levels: tuple[str, ...] = DEFAULT_BODY_PART_LEVELS
    pattern_levels: tuple[str, ...] = DEFAULT_PATTERN_LEVELS
    version: str = "xg_shot.v1"

    @classmethod
    def fit(
        cls,
        shots: pd.DataFrame,
        *,
        body_part_levels: tuple[str, ...] = DEFAULT_BODY_PART_LEVELS,
        pattern_levels: tuple[str, ...] = DEFAULT_PATTERN_LEVELS,
    ) -> XgShotModel:
        """Fit a logistic regression on ``shots``.

        ``shots`` must have columns: distance_to_goal, angle_to_goal, body_part,
        pattern_of_play, is_goal. Rows lacking any of those are dropped.
        """
        required = {
            "distance_to_goal",
            "angle_to_goal",
            "body_part",
            "pattern_of_play",
            "is_goal",
        }
        missing = required - set(shots.columns)
        if missing:
            raise ValueError(f"xG fit requires columns {missing} (missing)")
        df = shots.dropna(subset=list(required)).copy()
        if df.empty:
            raise ValueError("no usable shots after dropping NaNs")
        feature_names, X = _build_feature_matrix(
            df, body_part_levels=body_part_levels, pattern_levels=pattern_levels
        )
        y = df["is_goal"].astype(int).to_numpy()
        # L2 regularisation with C=1.0 is the standard default — we leave the
        # penalty kwarg unset because sklearn 1.8 deprecated explicit values.
        clf = LogisticRegression(C=1.0, max_iter=500)
        clf.fit(X, y)
        return cls(
            intercept=float(clf.intercept_[0]),
            feature_names=tuple(feature_names),
            coefficients=tuple(float(c) for c in clf.coef_[0]),
            body_part_levels=body_part_levels,
            pattern_levels=pattern_levels,
        )

    def predict_proba(self, shots: pd.DataFrame) -> np.ndarray:
        """Return p(goal | shot) for every row in ``shots``.

        Missing categorical values are mapped to the reference level (so the
        prediction is the intercept-anchored baseline for that geometry).
        """
        _, X = _build_feature_matrix(
            shots,
            body_part_levels=self.body_part_levels,
            pattern_levels=self.pattern_levels,
            template_features=self.feature_names,
        )
        coef = np.asarray(self.coefficients)
        logits = self.intercept + X @ coef
        return _sigmoid(logits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "intercept": self.intercept,
            "feature_names": list(self.feature_names),
            "coefficients": list(self.coefficients),
            "body_part_levels": list(self.body_part_levels),
            "pattern_levels": list(self.pattern_levels),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> XgShotModel:
        return cls(
            intercept=float(payload["intercept"]),
            feature_names=tuple(payload["feature_names"]),
            coefficients=tuple(float(c) for c in payload["coefficients"]),
            body_part_levels=tuple(payload.get("body_part_levels", DEFAULT_BODY_PART_LEVELS)),
            pattern_levels=tuple(payload.get("pattern_levels", DEFAULT_PATTERN_LEVELS)),
            version=str(payload.get("version", "xg_shot.v1")),
        )

    def save(self, path: Path = DEFAULT_ARTIFACT_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path = DEFAULT_ARTIFACT_PATH) -> XgShotModel:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _build_feature_matrix(
    df: pd.DataFrame,
    *,
    body_part_levels: tuple[str, ...],
    pattern_levels: tuple[str, ...],
    template_features: tuple[str, ...] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Construct the (feature_names, X) tuple used for fit and predict.

    When ``template_features`` is provided, columns are reordered/restricted
    to match it exactly; that's the path taken at prediction time so the
    matrix layout matches the persisted coefficients regardless of how the
    input DataFrame was constructed.
    """
    distance = pd.to_numeric(df["distance_to_goal"], errors="coerce").to_numpy(dtype=float)
    angle = pd.to_numeric(df["angle_to_goal"], errors="coerce").to_numpy(dtype=float)
    # inverse_distance is a classic xG feature (Caley 2015) — saturates the
    # goal-mouth advantage so a 2m chance and a 1m chance are differentiable.
    inv_distance = 1.0 / (1.0 + distance)
    features: dict[str, np.ndarray] = {
        "distance_to_goal": distance,
        "angle_to_goal": angle,
        "inv_distance": inv_distance,
    }
    body_part = df["body_part"].fillna(REFERENCE_BODY_PART)
    for level in body_part_levels:
        if level == REFERENCE_BODY_PART:
            continue
        col = f"bp_{level.lower().replace(' ', '_')}"
        features[col] = (body_part == level).to_numpy(dtype=float)
    pattern = df["pattern_of_play"].fillna(REFERENCE_PATTERN)
    for level in pattern_levels:
        if level == REFERENCE_PATTERN:
            continue
        col = f"pat_{level.lower().replace(' ', '_')}"
        features[col] = (pattern == level).to_numpy(dtype=float)
    if template_features is not None:
        ordered_names = list(template_features)
        cols = []
        n = len(df)
        for name in ordered_names:
            if name in features:
                cols.append(features[name])
            else:
                cols.append(np.zeros(n, dtype=float))
    else:
        ordered_names = list(features.keys())
        cols = [features[name] for name in ordered_names]
    X = np.column_stack(cols) if cols else np.zeros((len(df), 0))
    return ordered_names, X


def fit_and_save(
    shots: pd.DataFrame,
    *,
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
) -> XgShotModel:
    """Fit + persist in one call. Returns the fitted model."""
    model = XgShotModel.fit(shots)
    model.save(artifact_path)
    return model


def attach_predicted_xg(
    shots: pd.DataFrame,
    model: XgShotModel,
    *,
    column: str = "our_xg",
) -> pd.DataFrame:
    """Return a copy of ``shots`` with a ``column`` column holding p(goal | shot)."""
    out = shots.copy()
    out[column] = model.predict_proba(shots)
    return out


__all__ = [
    "DEFAULT_ARTIFACT_PATH",
    "DEFAULT_BODY_PART_LEVELS",
    "DEFAULT_PATTERN_LEVELS",
    "REFERENCE_BODY_PART",
    "REFERENCE_PATTERN",
    "XgShotModel",
    "attach_predicted_xg",
    "fit_and_save",
]
