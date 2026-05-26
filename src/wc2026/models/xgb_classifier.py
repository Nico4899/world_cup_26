"""XGBoost H/D/A classifier on the materialised match-feature table.

This is Phase 5's secondary engine — the structural Poisson model remains the
primary; the XGBoost output is blended (geometric mean, see ``models.blend``)
to produce the final 1X2 probabilities. The design follows Groll, Ley,
Schauberger & Van Eetvelde 2019: take the Poisson team-ability outputs as
additional features for a flexible learner, rather than discarding them.

Features expected (from ``db.models.MatchFeatures``):
    elo_diff, fifa_rank_diff, xg_form_diff, rest_days_diff, squad_age_diff,
    is_neutral, is_host_home, is_host_away,
    poisson_exp_home_goals, poisson_exp_away_goals,
    poisson_p_home, poisson_p_draw, poisson_p_away

XGBoost natively handles NaN; we deliberately do *not* impute missing values
in this module — the tree learns its own missing-value branch per split.

Persistence: the trained booster is saved as XGBoost's JSON format alongside
a tiny sidecar JSON listing the feature column names and a model-version tag.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# Feature columns the model consumes, in the canonical order they were trained
# in. Stored on the artefact so prediction-time reordering is automatic.
#
# Wave 2 grew this list from 13 to 14: ``venue_altitude_m`` lands here so
# the next ``xgb_refit`` picks it up. ``predict_proba`` keys off the loaded
# artifact's own feature_names — older artifacts still work, they simply
# ignore the new column. ``venue_wet_bulb_c`` is intentionally NOT in the
# default training list yet (its predictive value on past tournaments is
# uncertain pending the held-out backtest gate); it's persisted to the
# table so future retrains can opt in by passing ``feature_names=`` with
# the extended tuple.
DEFAULT_FEATURE_COLUMNS: tuple[str, ...] = (
    "elo_diff",
    "fifa_rank_diff",
    "xg_form_diff",
    "rest_days_diff",
    "squad_age_diff",
    "is_neutral",
    "is_host_home",
    "is_host_away",
    "poisson_exp_home_goals",
    "poisson_exp_away_goals",
    "poisson_p_home",
    "poisson_p_draw",
    "poisson_p_away",
    "venue_altitude_m",
)

# Class encoding — order matches the (home_win, draw, away_win) tuple every
# other module uses, so callers can read predict_proba[:, 0/1/2] without an
# extra permutation step.
CLASS_HOME = 0
CLASS_DRAW = 1
CLASS_AWAY = 2
N_CLASSES = 3

DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "objective": "multi:softprob",
    "num_class": N_CLASSES,
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "reg_lambda": 1.0,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    # tree_method=hist is the modern default and handles NaN natively.
    "tree_method": "hist",
}

DEFAULT_ARTIFACT_DIR = Path("data/artifacts/xgb")
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACT_DIR / "latest.json"
DEFAULT_META_PATH = DEFAULT_ARTIFACT_DIR / "latest_meta.json"


@dataclass
class XgbMatchModel:
    """Trained XGB H/D/A classifier wrapper.

    Holds the booster, the canonical feature order, hyperparams, and a version
    string. Constructed via ``XgbMatchModel.fit(...)`` or ``XgbMatchModel.load(...)``.
    """

    booster: XGBClassifier
    feature_names: tuple[str, ...]
    version: str = "xgb_match.v1"
    hyperparams: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        *,
        sample_weight: np.ndarray | pd.Series | None = None,
        feature_names: tuple[str, ...] | None = None,
        hyperparams: dict[str, Any] | None = None,
        random_state: int = 0,
    ) -> XgbMatchModel:
        """Fit an XGB H/D/A classifier on ``X`` (one row per match).

        ``X`` may contain additional columns — only the ones in
        ``feature_names`` (default: DEFAULT_FEATURE_COLUMNS) are used. ``y``
        must be integer class labels in {0=H, 1=D, 2=A}.
        """
        cols = feature_names or DEFAULT_FEATURE_COLUMNS
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"X is missing required feature columns: {missing}")
        params = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
        params.setdefault("random_state", random_state)
        clf = XGBClassifier(**params)
        # Force every feature column to float64 so Python ``None``s in the source
        # DataFrames (which produce ``object`` dtype) become ``NaN`` — XGBoost
        # then takes its own missing-value branch instead of rejecting the input.
        X_numeric = X[list(cols)].apply(pd.to_numeric, errors="coerce").astype(float)
        clf.fit(
            X_numeric,
            np.asarray(y, dtype=int),
            sample_weight=None if sample_weight is None else np.asarray(sample_weight, dtype=float),
        )
        return cls(booster=clf, feature_names=tuple(cols), hyperparams=params)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return an ``(n, 3)`` probability matrix in (home, draw, away) order.

        Missing feature columns become a NaN-filled column so XGBoost takes its
        learned missing-value branch — predictions still run end-to-end even
        when an upstream source is offline.
        """
        df = X.copy()
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = np.nan
        ordered = df[list(self.feature_names)].apply(pd.to_numeric, errors="coerce").astype(float)
        return self.booster.predict_proba(ordered)

    def save(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        meta_path: Path = DEFAULT_META_PATH,
    ) -> tuple[Path, Path]:
        """Persist the booster + the feature/meta sidecar. Returns (model, meta) paths."""
        model_path = Path(model_path)
        meta_path = Path(meta_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(model_path))
        meta_path.write_text(
            json.dumps(
                {
                    "version": self.version,
                    "feature_names": list(self.feature_names),
                    "hyperparams": self.hyperparams,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return model_path, meta_path

    @classmethod
    def load(
        cls,
        model_path: Path = DEFAULT_MODEL_PATH,
        meta_path: Path = DEFAULT_META_PATH,
    ) -> XgbMatchModel:
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        feature_names = tuple(meta["feature_names"])
        hyperparams = meta.get("hyperparams", {})
        # Reconstruct a classifier shell so XGBClassifier's predict_proba works
        # after load_model (the wrapper needs n_classes_ wired through).
        clf = XGBClassifier(**hyperparams)
        clf.load_model(str(model_path))
        return cls(
            booster=clf,
            feature_names=feature_names,
            version=str(meta.get("version", "xgb_match.v1")),
            hyperparams=hyperparams,
        )


def label_from_score(home_score: int, away_score: int) -> int:
    """Encode a final-time scoreline as ``CLASS_HOME`` / ``CLASS_DRAW`` / ``CLASS_AWAY``."""
    if home_score > away_score:
        return CLASS_HOME
    if home_score < away_score:
        return CLASS_AWAY
    return CLASS_DRAW


def labels_for_matches(matches: pd.DataFrame) -> np.ndarray:
    """Vectorised version of :func:`label_from_score` over a played-matches df.

    Expects integer ``home_score`` / ``away_score`` columns; rows with NaN
    scores raise ``ValueError`` (training on a row with no outcome is a bug).
    """
    if {"home_score", "away_score"} - set(matches.columns):
        raise ValueError("labels_for_matches requires home_score + away_score columns")
    h = pd.to_numeric(matches["home_score"], errors="coerce")
    a = pd.to_numeric(matches["away_score"], errors="coerce")
    if h.isna().any() or a.isna().any():
        raise ValueError("labels_for_matches got rows with NaN scores")
    labels = np.where(
        h.to_numpy() > a.to_numpy(),
        CLASS_HOME,
        np.where(h.to_numpy() < a.to_numpy(), CLASS_AWAY, CLASS_DRAW),
    )
    return labels.astype(int)


__all__ = [
    "CLASS_AWAY",
    "CLASS_DRAW",
    "CLASS_HOME",
    "DEFAULT_FEATURE_COLUMNS",
    "DEFAULT_HYPERPARAMS",
    "DEFAULT_META_PATH",
    "DEFAULT_MODEL_PATH",
    "N_CLASSES",
    "XgbMatchModel",
    "label_from_score",
    "labels_for_matches",
]
