"""SHAP wrapper around the XGB H/D/A classifier.

Surfaces the per-prediction feature contributions for one outcome class
(default: ``CLASS_HOME``) — the dashboard's "why this prediction" panel
displays the top-N features by absolute SHAP value.

We use ``shap.TreeExplainer``, which is exact for tree-ensemble models and
cheap to evaluate (microseconds per row). Output is a list of
``{feature, value, contribution}`` dicts so the API can return JSON without
NumPy-dependent client code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap

from wc2026.models.xgb_classifier import (
    CLASS_AWAY,
    CLASS_DRAW,
    CLASS_HOME,
    XgbMatchModel,
)

# Mapping for human-readable class labels in API responses / dashboard tooltips.
CLASS_NAMES: dict[int, str] = {
    CLASS_HOME: "home_win",
    CLASS_DRAW: "draw",
    CLASS_AWAY: "away_win",
}


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    value: float | None  # the input value (None if NaN)
    contribution: float  # signed SHAP value


@dataclass
class XgbExplainer:
    """Thin wrapper bundling the booster + an attached SHAP TreeExplainer."""

    model: XgbMatchModel
    explainer: shap.TreeExplainer

    @classmethod
    def from_model(cls, model: XgbMatchModel) -> XgbExplainer:
        return cls(
            model=model,
            explainer=shap.TreeExplainer(model.booster, feature_perturbation="tree_path_dependent"),
        )

    def explain_row(
        self,
        features: pd.DataFrame | dict[str, Any],
        *,
        class_index: int = CLASS_HOME,
    ) -> list[FeatureContribution]:
        """Per-feature SHAP values for one row, sorted by |contribution| desc."""
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        if class_index not in CLASS_NAMES:
            raise ValueError(f"class_index must be in {sorted(CLASS_NAMES)}; got {class_index}")
        # Reorder columns to match training, NaN-fill missing ones.
        df = features.copy()
        for col in self.model.feature_names:
            if col not in df.columns:
                df[col] = np.nan
        ordered = (
            df[list(self.model.feature_names)].apply(pd.to_numeric, errors="coerce").astype(float)
        )
        sv = self.explainer.shap_values(ordered)
        # TreeExplainer on a multi-class model returns shape (n, n_features, n_classes)
        # in modern shap; older shap returned a list. Be tolerant.
        sv_for_class = _select_class_shap(sv, class_index)
        # We only ever explain one row at a time on this code path.
        row_shap = sv_for_class[0]
        rows: list[FeatureContribution] = []
        for name, contribution in zip(self.model.feature_names, row_shap, strict=True):
            raw = ordered.iloc[0][name]
            value = None if (isinstance(raw, float) and np.isnan(raw)) else raw
            rows.append(
                FeatureContribution(
                    feature=name,
                    value=None if value is None else float(value),
                    contribution=float(contribution),
                )
            )
        rows.sort(key=lambda r: abs(r.contribution), reverse=True)
        return rows

    def top_features(
        self,
        features: pd.DataFrame | dict[str, Any],
        *,
        class_index: int = CLASS_HOME,
        n: int = 5,
    ) -> list[FeatureContribution]:
        """Convenience: top-``n`` features by absolute SHAP contribution."""
        return self.explain_row(features, class_index=class_index)[:n]


def _select_class_shap(sv: Any, class_index: int) -> np.ndarray:
    """Pull the per-class slice out of whatever shape ``sv`` ended up in.

    SHAP's TreeExplainer.shap_values returns:
        * a list of length n_classes (legacy)
        * an ndarray of shape (n_samples, n_features, n_classes) (modern)
        * an ndarray of shape (n_samples, n_features) for binary classifiers
    """
    if isinstance(sv, list):
        return np.asarray(sv[class_index])
    arr = np.asarray(sv)
    if arr.ndim == 3:
        return arr[..., class_index]
    if arr.ndim == 2:
        # Binary / single-class case — the caller is asking for the only class.
        return arr
    raise ValueError(f"Unexpected SHAP shape: {arr.shape}")


__all__ = [
    "CLASS_NAMES",
    "FeatureContribution",
    "XgbExplainer",
]
