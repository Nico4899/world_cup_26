"""Refit the Phase 5 XGB H/D/A classifier on the historical match corpus.

The pipeline:

1. Load played matches (Jürisoo)
2. Filter to a training window (default: last 10 years from ``ref_date``)
3. Fit PoissonDC on the *training* matches only
4. Build features for every training row using that PoissonDC
5. Compute sample weights via ``features.match_weights.combined_weight``
6. Fit XGB H/D/A on (features, labels)
7. Persist to ``data/artifacts/xgb/{latest.json, latest_meta.json}``

The hindcast-gate helper (:func:`hindcast_gate`) reproduces this end-to-end on
WC 2022 with a strict pre-tournament cutoff and reports both Poisson-only and
blended log-loss / Brier / RPS, used by Phase 5.8 to verify the blended model
doesn't regress.

CLI usage::

    uv run python scripts/refit_xgb.py            # refit, save artefact
    uv run python scripts/refit_xgb.py --hindcast # run the WC 2022 gate
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from wc2026.eval.calibration import (
    baseline_log_loss,
    match_brier,
    match_log_loss,
    match_rps,
)
from wc2026.features.build_match_features import (
    FeatureSources,
    MatchSpec,
    build_features_for_matches,
)
from wc2026.features.match_weights import combined_weight
from wc2026.ingest.kaggle_intl import load_played
from wc2026.models.blend import blend_geometric
from wc2026.models.poisson_dc import PoissonDC
from wc2026.models.xgb_classifier import (
    CLASS_AWAY,
    CLASS_DRAW,
    CLASS_HOME,
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_META_PATH,
    DEFAULT_MODEL_PATH,
    XgbMatchModel,
    labels_for_matches,
)

DEFAULT_HALF_LIFE_DAYS = 3650  # 10 years — matches scripts/refit_poisson_dc.py
DEFAULT_HISTORY_YEARS = 10
WC_2022_OPENER = pd.Timestamp("2022-11-20")
WC_2022_FINAL = pd.Timestamp("2022-12-18")

logger = logging.getLogger(__name__)


# ---- training corpus -------------------------------------------------------


@dataclass(frozen=True)
class CorpusResult:
    """Bundle of (X, y, sample_weight) plus diagnostic metadata."""

    features: pd.DataFrame
    labels: np.ndarray
    sample_weight: np.ndarray
    ref_date: pd.Timestamp
    n_matches: int
    poisson_model: PoissonDC


def _select_training_window(
    matches: pd.DataFrame,
    *,
    ref_date: pd.Timestamp,
    history_years: int,
) -> pd.DataFrame:
    cutoff = ref_date - pd.Timedelta(days=int(history_years * 365.25))
    df = matches[matches["date"] >= cutoff].copy()
    return df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)


def _specs_from_matches(matches: pd.DataFrame) -> list[MatchSpec]:
    return [
        MatchSpec(
            match_date=row["date"].date() if hasattr(row["date"], "date") else row["date"],
            home_team=row["home_team"],
            away_team=row["away_team"],
            neutral=bool(row.get("neutral", False)),
        )
        for _, row in matches.iterrows()
    ]


def build_training_corpus(
    *,
    ref_date: pd.Timestamp | None = None,
    history_years: int = DEFAULT_HISTORY_YEARS,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    upper_cutoff: pd.Timestamp | None = None,
) -> CorpusResult:
    """Produce (X, y, sample_weight) for XGB training.

    Set ``upper_cutoff`` to exclude matches on or after that date — used by the
    hindcast gate to leave a clean test window.
    """
    ref_date = ref_date or pd.Timestamp(datetime.now(UTC).date())
    all_played = load_played()
    df = _select_training_window(all_played, ref_date=ref_date, history_years=history_years)
    if upper_cutoff is not None:
        df = df[df["date"] < upper_cutoff].reset_index(drop=True)
    if df.empty:
        raise ValueError("training window is empty — check ref_date / history_years")
    # Fit PoissonDC on the training window using the same combined weights
    # the XGB will see. This is "self-leakage" inside the training set but is
    # not a leakage of the test set.
    weights = combined_weight(df, ref_date=ref_date, half_life_days=half_life_days)
    poisson = PoissonDC().fit(df, weights=weights.to_numpy())
    # Build feature rows.
    sources = FeatureSources(
        matches=all_played,  # historical matches for rest-days lookup
        poisson_model=poisson,
        snapshot_meta={
            "training_ref_date": ref_date.date().isoformat(),
            "training_history_years": history_years,
        },
    )
    specs = _specs_from_matches(df)
    features_df = build_features_for_matches(specs, sources)
    features_only = features_df[list(DEFAULT_FEATURE_COLUMNS)]
    labels = labels_for_matches(df)
    return CorpusResult(
        features=features_only,
        labels=labels,
        sample_weight=weights.to_numpy(),
        ref_date=ref_date,
        n_matches=len(df),
        poisson_model=poisson,
    )


# ---- training + persistence ------------------------------------------------


def refit_and_save(
    *,
    ref_date: pd.Timestamp | None = None,
    history_years: int = DEFAULT_HISTORY_YEARS,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    model_path: Path = DEFAULT_MODEL_PATH,
    meta_path: Path = DEFAULT_META_PATH,
) -> Path:
    """Refit XGB on a fresh training corpus and persist. Returns model path."""
    corpus = build_training_corpus(
        ref_date=ref_date,
        history_years=history_years,
        half_life_days=half_life_days,
    )
    model = XgbMatchModel.fit(
        corpus.features,
        corpus.labels,
        sample_weight=corpus.sample_weight,
    )
    out_model, _ = model.save(model_path, meta_path)
    logger.info(
        "xgb refit: trained on %d matches; ref_date=%s; saved to %s",
        corpus.n_matches,
        corpus.ref_date.date(),
        out_model,
    )
    return out_model


# ---- hindcast gate ---------------------------------------------------------


@dataclass(frozen=True)
class HindcastResult:
    n_test_matches: int
    poisson_only_log_loss: float
    poisson_only_brier: float
    poisson_only_rps: float
    blended_log_loss: float
    blended_brier: float
    blended_rps: float
    climatological_log_loss: float


def _poisson_probs_for_specs(model: PoissonDC, specs: list[MatchSpec]) -> np.ndarray:
    """Vectorised PoissonDC 1X2 probabilities for a list of specs."""
    rows = np.zeros((len(specs), 3), dtype=float)
    for i, s in enumerate(specs):
        try:
            outcomes = model.outcome_probs(s.home_team, s.away_team, neutral=s.neutral)
        except (KeyError, ValueError):
            # Missing team → uniform; happens for tiny associations.
            rows[i] = (1 / 3, 1 / 3, 1 / 3)
            continue
        rows[i] = (outcomes["home_win"], outcomes["draw"], outcomes["away_win"])
    return rows


def hindcast_gate(
    *,
    cutoff: pd.Timestamp = WC_2022_OPENER,
    final: pd.Timestamp = WC_2022_FINAL,
    history_years: int = DEFAULT_HISTORY_YEARS,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    blend_weight: float = 0.5,
) -> HindcastResult:
    """End-to-end no-leakage hindcast on WC 2022.

    Trains XGB on matches strictly < ``cutoff``, evaluates on WC 2022 matches
    in the [cutoff, final] window. Returns calibration for Poisson-only and
    for the geometric blend.
    """
    all_played = load_played()
    test = all_played[
        (all_played["date"] >= cutoff)
        & (all_played["date"] <= final)
        & (all_played["tournament"] == "FIFA World Cup")
    ].copy()
    if test.empty:
        raise ValueError("hindcast_gate: no WC 2022 matches in the test window — corpus missing?")
    corpus = build_training_corpus(
        ref_date=cutoff,
        history_years=history_years,
        half_life_days=half_life_days,
        upper_cutoff=cutoff,
    )
    model = XgbMatchModel.fit(
        corpus.features,
        corpus.labels,
        sample_weight=corpus.sample_weight,
    )

    test_specs = _specs_from_matches(test)
    test_features_df = build_features_for_matches(
        test_specs,
        FeatureSources(
            matches=all_played,
            poisson_model=corpus.poisson_model,
        ),
    )
    poisson_probs = _poisson_probs_for_specs(corpus.poisson_model, test_specs)
    xgb_probs = model.predict_proba(test_features_df[list(DEFAULT_FEATURE_COLUMNS)])
    blended_probs = blend_geometric(poisson_probs, xgb_probs, weight=blend_weight)
    labels = labels_for_matches(test)

    outcome_strings = [
        {CLASS_HOME: "H", CLASS_DRAW: "D", CLASS_AWAY: "A"}[int(label)] for label in labels
    ]
    poisson_eval = _evaluate_calibration(poisson_probs, outcome_strings)
    blend_eval = _evaluate_calibration(blended_probs, outcome_strings)
    clim = baseline_log_loss(outcome_strings)
    return HindcastResult(
        n_test_matches=len(test),
        poisson_only_log_loss=poisson_eval["log_loss"],
        poisson_only_brier=poisson_eval["brier"],
        poisson_only_rps=poisson_eval["rps"],
        blended_log_loss=blend_eval["log_loss"],
        blended_brier=blend_eval["brier"],
        blended_rps=blend_eval["rps"],
        climatological_log_loss=clim,
    )


def _evaluate_calibration(probs: np.ndarray, outcome_strings: list[str]) -> dict[str, float]:
    log_losses, briers, rpses = [], [], []
    for i, observed in enumerate(outcome_strings):
        p_home, p_draw, p_away = (float(x) for x in probs[i])
        log_losses.append(match_log_loss(observed, p_home, p_draw, p_away))
        briers.append(match_brier(observed, p_home, p_draw, p_away))
        rpses.append(match_rps(observed, p_home, p_draw, p_away))
    return {
        "log_loss": float(np.mean(log_losses)),
        "brier": float(np.mean(briers)),
        "rps": float(np.mean(rpses)),
    }


# ---- CLI -------------------------------------------------------------------


def _format_hindcast(result: HindcastResult) -> str:
    return (
        f"WC 2022 hindcast ({result.n_test_matches} matches):\n"
        f"  poisson-only:  log-loss={result.poisson_only_log_loss:.4f}  "
        f"brier={result.poisson_only_brier:.4f}  rps={result.poisson_only_rps:.4f}\n"
        f"  blended (w=0.5): log-loss={result.blended_log_loss:.4f}  "
        f"brier={result.blended_brier:.4f}  rps={result.blended_rps:.4f}\n"
        f"  climatological log-loss = {result.climatological_log_loss:.4f}\n"
        f"  Delta (blend - poisson) log-loss = "
        f"{result.blended_log_loss - result.poisson_only_log_loss:+.4f} "
        f"(negative = blend wins)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hindcast",
        action="store_true",
        help="Run the WC 2022 no-leakage gate instead of refitting.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.hindcast:
        result = hindcast_gate()
        print(_format_hindcast(result))
        return
    refit_and_save()


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_HALF_LIFE_DAYS",
    "DEFAULT_HISTORY_YEARS",
    "WC_2022_FINAL",
    "WC_2022_OPENER",
    "CorpusResult",
    "HindcastResult",
    "build_training_corpus",
    "hindcast_gate",
    "refit_and_save",
]
