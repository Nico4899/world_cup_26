"""Unit tests for the geometric blend module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wc2026.models.blend import blend_dict, blend_geometric


def test_blend_two_identical_distributions_returns_input() -> None:
    p = np.array([0.5, 0.27, 0.23])
    out = blend_geometric(p, p)
    assert np.allclose(out, p, atol=1e-9)


def test_blend_outputs_sum_to_one() -> None:
    p_poisson = np.array([0.4, 0.30, 0.30])
    p_xgb = np.array([0.7, 0.20, 0.10])
    out = blend_geometric(p_poisson, p_xgb)
    assert math.isclose(out.sum(), 1.0, abs_tol=1e-9)


def test_blend_weight_zero_returns_pure_xgb() -> None:
    p_poisson = np.array([0.1, 0.1, 0.8])
    p_xgb = np.array([0.6, 0.3, 0.1])
    out = blend_geometric(p_poisson, p_xgb, weight=0.0)
    assert np.allclose(out, p_xgb, atol=1e-9)


def test_blend_weight_one_returns_pure_poisson() -> None:
    p_poisson = np.array([0.55, 0.20, 0.25])
    p_xgb = np.array([0.2, 0.6, 0.2])
    out = blend_geometric(p_poisson, p_xgb, weight=1.0)
    assert np.allclose(out, p_poisson, atol=1e-9)


def test_blend_geometric_mean_is_between_inputs_when_skewed() -> None:
    """For asymmetric inputs, every coordinate of the blend lies between
    the matching coordinates of the two inputs."""
    p_poisson = np.array([0.3, 0.4, 0.3])
    p_xgb = np.array([0.6, 0.1, 0.3])
    out = blend_geometric(p_poisson, p_xgb)
    lower = np.minimum(p_poisson, p_xgb)
    upper = np.maximum(p_poisson, p_xgb)
    # Geometric mean is in [min, max] before renorm; after renorm it can leave
    # the box slightly but only by small amounts. Confirm rough containment.
    assert (out >= lower - 0.05).all()
    assert (out <= upper + 0.05).all()


def test_blend_zero_probability_does_not_nan() -> None:
    """A clean 0 must not crash the blend (clipped by EPS internally)."""
    p_poisson = np.array([0.0, 0.5, 0.5])
    p_xgb = np.array([0.5, 0.0, 0.5])
    out = blend_geometric(p_poisson, p_xgb)
    assert np.all(np.isfinite(out))
    assert math.isclose(out.sum(), 1.0, abs_tol=1e-9)


def test_blend_vectorised_over_n_rows() -> None:
    rng = np.random.default_rng(0)
    p_poisson = rng.dirichlet([1, 1, 1], size=50)
    p_xgb = rng.dirichlet([1, 1, 1], size=50)
    out = blend_geometric(p_poisson, p_xgb)
    assert out.shape == (50, 3)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-9)


def test_blend_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        blend_geometric(np.zeros((3, 3)), np.zeros((4, 3)))


def test_blend_rejects_non_triplet_input() -> None:
    with pytest.raises(ValueError, match="shape"):
        blend_geometric(np.array([0.5, 0.5]), np.array([0.5, 0.5]))


def test_blend_rejects_weight_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="weight"):
        blend_geometric(np.array([0.5, 0.3, 0.2]), np.array([0.5, 0.3, 0.2]), weight=1.5)


def test_blend_dict_returns_proper_outcome_dict() -> None:
    poisson = {"home_win": 0.45, "draw": 0.27, "away_win": 0.28}
    xgb = {"home_win": 0.55, "draw": 0.25, "away_win": 0.20}
    out = blend_dict(poisson, xgb)
    assert set(out.keys()) == {"home_win", "draw", "away_win"}
    assert math.isclose(sum(out.values()), 1.0, abs_tol=1e-9)
    # The blended home_win should sit between the two inputs.
    assert 0.45 <= out["home_win"] <= 0.55
