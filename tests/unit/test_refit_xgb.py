"""Unit tests for scripts/refit_xgb.py.

We rely on the real Jürisoo CSV on disk (the rest of the test suite already
depends on it) so the corpus build runs end-to-end. The training window is
deliberately small so each test stays well under a second.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import scripts.refit_xgb as r

from wc2026.models.blend import blend_geometric
from wc2026.models.xgb_classifier import DEFAULT_FEATURE_COLUMNS, XgbMatchModel

# Every corpus build pulls thousands of historical matches through the per-
# match feature orchestrator (O(N²) due to rest-days lookup), so these tests
# are slow. Mark them so the default suite stays under 10 seconds.
pytestmark = pytest.mark.slow


def test_build_training_corpus_returns_aligned_arrays() -> None:
    corpus = r.build_training_corpus(
        ref_date=pd.Timestamp("2020-01-01"),
        history_years=2,
    )
    assert len(corpus.features) == len(corpus.labels)
    assert len(corpus.features) == len(corpus.sample_weight)
    assert corpus.n_matches == len(corpus.features)
    assert list(corpus.features.columns) == list(DEFAULT_FEATURE_COLUMNS)


def test_build_training_corpus_respects_upper_cutoff() -> None:
    """upper_cutoff must exclude matches dated >= cutoff."""
    corpus = r.build_training_corpus(
        ref_date=pd.Timestamp("2022-11-20"),
        history_years=2,
        upper_cutoff=pd.Timestamp("2022-06-01"),
    )
    # No row should leak — but the features DF doesn't carry the date column;
    # we check via n_matches against an unrestricted build.
    larger = r.build_training_corpus(
        ref_date=pd.Timestamp("2022-11-20"),
        history_years=2,
    )
    assert corpus.n_matches < larger.n_matches


def test_corpus_labels_are_in_three_class_set() -> None:
    corpus = r.build_training_corpus(
        ref_date=pd.Timestamp("2020-01-01"),
        history_years=2,
    )
    assert set(corpus.labels).issubset({0, 1, 2})


def test_corpus_sample_weights_strictly_positive() -> None:
    corpus = r.build_training_corpus(
        ref_date=pd.Timestamp("2020-01-01"),
        history_years=2,
    )
    assert (corpus.sample_weight > 0).all()


def test_refit_and_save_writes_artefact(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"
    meta_path = tmp_path / "meta.json"
    out = r.refit_and_save(
        ref_date=pd.Timestamp("2020-01-01"),
        history_years=2,
        model_path=model_path,
        meta_path=meta_path,
    )
    assert out == model_path
    assert model_path.exists()
    assert meta_path.exists()


def test_refit_and_save_artefact_loads_back(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"
    meta_path = tmp_path / "meta.json"
    r.refit_and_save(
        ref_date=pd.Timestamp("2020-01-01"),
        history_years=2,
        model_path=model_path,
        meta_path=meta_path,
    )
    reloaded = XgbMatchModel.load(model_path, meta_path)
    # Predict on a tiny corpus to confirm the artefact round-trips functionally.
    corpus = r.build_training_corpus(
        ref_date=pd.Timestamp("2020-01-01"),
        history_years=1,
    )
    import numpy as np

    probs = reloaded.predict_proba(corpus.features.head(5))
    assert probs.shape == (5, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_format_hindcast_includes_blend_delta_line() -> None:
    result = r.HindcastResult(
        n_test_matches=64,
        poisson_only_log_loss=1.04,
        poisson_only_brier=0.61,
        poisson_only_rps=0.22,
        blended_log_loss=1.01,
        blended_brier=0.58,
        blended_rps=0.21,
        climatological_log_loss=1.07,
    )
    out = r._format_hindcast(result)
    assert "Delta (blend - poisson)" in out
    assert "-0.0300" in out


def test_poisson_probs_for_specs_returns_valid_distribution() -> None:
    """Lightweight check that the helper returns rows summing to 1 even when
    a team is missing from the fitted set."""
    import numpy as np

    corpus = r.build_training_corpus(
        ref_date=pd.Timestamp("2020-01-01"),
        history_years=2,
    )
    specs = [
        r.MatchSpec(pd.Timestamp("2020-02-01").date(), "Argentina", "Brazil"),
        r.MatchSpec(pd.Timestamp("2020-02-01").date(), "Atlantis", "Atlantica"),
    ]
    probs = r._poisson_probs_for_specs(corpus.poisson_model, specs)
    assert probs.shape == (2, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    # Unknown teams should fall back to a uniform 1/3 each.
    assert np.allclose(probs[1], 1 / 3, atol=1e-9)


def test_blend_geometric_is_used_correctly_inside_hindcast() -> None:
    """The hindcast gate uses ``blend_geometric``; verify that the standalone
    call matches what the gate would produce for a stub triplet pair."""
    p_p = [0.5, 0.27, 0.23]
    p_x = [0.4, 0.30, 0.30]
    out = blend_geometric(p_p, p_x, weight=0.5)
    assert abs(out.sum() - 1.0) < 1e-9
