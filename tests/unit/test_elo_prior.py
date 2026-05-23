"""Unit tests for Elo prior features + PoissonDCWithPrior."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from wc2026.features.elo_features import elo_to_strength_prior
from wc2026.models.elo_prior import elo_prior_centres
from wc2026.models.poisson_dc import PoissonDC
from wc2026.models.poisson_dc_with_prior import (
    PoissonDCWithPrior,
    _penalised_objective,
)

# --- elo_features ----------------------------------------------------------


def test_strength_prior_sums_to_zero_when_centred() -> None:
    df = pd.DataFrame(
        {"team_name": ["A", "B", "C", "D"], "rating": [1800.0, 1900.0, 2000.0, 2100.0]}
    )
    out = elo_to_strength_prior(df)
    attacks = [a for a, _ in out.values()]
    defences = [d for _, d in out.values()]
    assert sum(attacks) == pytest.approx(0.0, abs=1e-9)
    assert sum(defences) == pytest.approx(0.0, abs=1e-9)
    # defence is negation of attack
    for a, d in out.values():
        assert d == pytest.approx(-a)


def test_strength_prior_scales_by_100() -> None:
    df = pd.DataFrame({"team_name": ["A", "B"], "rating": [1900.0, 2100.0]})
    out = elo_to_strength_prior(df)
    # mean = 2000; A: (1900-2000)/100 = -1.0; B: +1.0
    assert out["A"] == pytest.approx((-1.0, 1.0))
    assert out["B"] == pytest.approx((1.0, -1.0))


def test_strength_prior_empty_input_returns_empty_dict() -> None:
    df = pd.DataFrame({"team_name": [], "rating": []})
    assert elo_to_strength_prior(df) == {}


def test_strength_prior_missing_columns_raises() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        elo_to_strength_prior(pd.DataFrame({"team_name": ["A"]}))


# --- elo_prior_centres alignment -------------------------------------------


def test_elo_prior_centres_aligns_to_team_order() -> None:
    snap = pd.DataFrame({"team_name": ["X", "Y", "Z"], "rating": [1800.0, 1900.0, 2000.0]})
    # fit order is sorted alphabetically by convention
    teams = ["X", "Y", "Z"]
    atk, dfc = elo_prior_centres(teams, snap)
    assert atk.shape == (3,)
    assert dfc.shape == (3,)
    # mean 1900; centres = -1.0, 0.0, +1.0
    assert atk == pytest.approx([-1.0, 0.0, 1.0])
    assert dfc == pytest.approx([1.0, 0.0, -1.0])


def test_elo_prior_centres_missing_teams_get_zero() -> None:
    snap = pd.DataFrame({"team_name": ["X", "Y"], "rating": [1900.0, 2100.0]})
    teams = ["X", "Y", "Z"]  # Z absent from snapshot
    atk, dfc = elo_prior_centres(teams, snap)
    assert atk[2] == 0.0
    assert dfc[2] == 0.0


# --- penalised objective: gradient correctness ------------------------------


def _make_synth_problem(rng: np.random.Generator, n_teams: int = 6, n_matches: int = 300):
    h_idx = rng.integers(0, n_teams, n_matches).astype(np.int64)
    a_idx = rng.integers(0, n_teams, n_matches).astype(np.int64)
    same = h_idx == a_idx
    a_idx[same] = (a_idx[same] + 1) % n_teams
    return {
        "h_idx": h_idx,
        "a_idx": a_idx,
        "h_score": rng.integers(0, 5, n_matches).astype(int),
        "a_score": rng.integers(0, 5, n_matches).astype(int),
        "not_neutral": (rng.random(n_matches) > 0.3).astype(float),
        "w": rng.uniform(0.1, 2.0, n_matches),
        "n_teams": n_teams,
    }


def test_penalised_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(7)
    n_teams = 6
    kwargs = _make_synth_problem(rng, n_teams=n_teams)
    attack_centres = rng.normal(0.0, 0.4, n_teams)
    defence_centres = rng.normal(0.0, 0.4, n_teams)
    # Centring not required for the objective itself; the fit code aligns
    # however it likes — we just test gradient correctness here.

    n_free = 2 * (n_teams - 1) + 2
    theta = rng.normal(0.0, 0.2, n_free)
    theta[2 * (n_teams - 1)] = 0.4
    theta[2 * (n_teams - 1) + 1] = 0.05

    full_kwargs = dict(
        **kwargs,
        attack_centres=attack_centres,
        defence_centres=defence_centres,
        prior_strength=1.7,
    )
    _, grad_analytic = _penalised_objective(theta, **full_kwargs)

    def f_only(t: np.ndarray) -> float:
        return _penalised_objective(t, **full_kwargs)[0]

    grad_numeric = approx_fprime(theta, f_only, epsilon=1e-6)
    rel_err = np.abs(grad_analytic - grad_numeric) / (np.abs(grad_numeric) + 1e-6)
    assert rel_err.max() < 5e-3, (
        f"max rel err {rel_err.max():.2e}; analytic={grad_analytic[:5]}; numeric={grad_numeric[:5]}"
    )


def test_penalised_objective_zero_prior_matches_base_nll() -> None:
    from wc2026.models.poisson_dc import _neg_log_lik_and_grad

    rng = np.random.default_rng(11)
    n_teams = 5
    kwargs = _make_synth_problem(rng, n_teams=n_teams, n_matches=150)
    n_free = 2 * (n_teams - 1) + 2
    theta = rng.normal(0.0, 0.2, n_free)
    theta[2 * (n_teams - 1)] = 0.3
    theta[2 * (n_teams - 1) + 1] = 0.04

    nll_base, grad_base = _neg_log_lik_and_grad(theta, **kwargs)
    nll_pen, grad_pen = _penalised_objective(
        theta,
        **kwargs,
        attack_centres=np.ones(n_teams),
        defence_centres=np.ones(n_teams),
        prior_strength=0.0,
    )
    assert nll_pen == pytest.approx(nll_base, abs=1e-12)
    assert np.allclose(grad_pen, grad_base, atol=1e-12)


# --- parameter recovery -----------------------------------------------------


def _simulate_matches(
    *,
    teams: list[str],
    true_attack: np.ndarray,
    true_defence: np.ndarray,
    home_advantage: float,
    n_matches: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n = len(teams)
    rows = []
    for _ in range(n_matches):
        h, a = rng.choice(n, size=2, replace=False)
        neutral = bool(rng.random() < 0.4)
        lh = np.exp(true_attack[h] + true_defence[a] + (0.0 if neutral else home_advantage))
        la = np.exp(true_attack[a] + true_defence[h])
        h_goals = rng.poisson(lh)
        a_goals = rng.poisson(la)
        rows.append(
            {
                "home_team": teams[h],
                "away_team": teams[a],
                "home_score": int(h_goals),
                "away_score": int(a_goals),
                "neutral": neutral,
                "date": pd.Timestamp("2020-01-01"),
                "tournament": "Friendly",
            }
        )
    return pd.DataFrame(rows)


def test_strong_prior_recovers_centres_on_tiny_data() -> None:
    """With prior_strength large and few matches, fitted attack/defence stay close
    to the prior centres rather than the data MLE."""
    rng = np.random.default_rng(33)
    n_teams = 5
    teams = [f"T{i}" for i in range(n_teams)]
    # Centre attack/defence under sum-to-zero
    raw = rng.normal(0.0, 0.5, n_teams)
    attack_centres = raw - raw.mean()
    raw2 = rng.normal(0.0, 0.5, n_teams)
    defence_centres = raw2 - raw2.mean()
    # Tiny dataset, sampled from very different "true" params
    df = _simulate_matches(
        teams=teams,
        true_attack=np.zeros(n_teams),
        true_defence=np.zeros(n_teams),
        home_advantage=0.3,
        n_matches=40,
        rng=rng,
    )
    model = PoissonDCWithPrior(
        attack_centres=attack_centres,
        defence_centres=defence_centres,
        teams=teams,
        prior_strength=500.0,
    ).fit(df)
    # With such a strong prior, attack/defence should be pulled very close to
    # centres after re-aligning to the model's fitted team order.
    fit_teams = list(model.params_.teams)
    prior_idx = {t: i for i, t in enumerate(teams)}
    expected_attack = np.array([attack_centres[prior_idx[t]] for t in fit_teams])
    expected_defence = np.array([defence_centres[prior_idx[t]] for t in fit_teams])
    assert np.allclose(model.params_.attack, expected_attack, atol=0.05)
    assert np.allclose(model.params_.defence, expected_defence, atol=0.05)


def test_zero_prior_strength_matches_base_poisson_dc() -> None:
    """prior_strength=0 should reproduce the unpenalised PoissonDC fit closely."""
    rng = np.random.default_rng(55)
    n_teams = 5
    teams = [f"T{i}" for i in range(n_teams)]
    raw = rng.normal(0.0, 0.4, n_teams)
    true_attack = raw - raw.mean()
    raw2 = rng.normal(0.0, 0.4, n_teams)
    true_defence = raw2 - raw2.mean()
    df = _simulate_matches(
        teams=teams,
        true_attack=true_attack,
        true_defence=true_defence,
        home_advantage=0.25,
        n_matches=800,
        rng=rng,
    )
    base = PoissonDC().fit(df)
    pen = PoissonDCWithPrior(
        attack_centres=np.zeros(n_teams),
        defence_centres=np.zeros(n_teams),
        teams=teams,
        prior_strength=0.0,
    ).fit(df)
    # Same fitted team order both ways
    assert base.params_.teams == pen.params_.teams
    assert np.allclose(base.params_.attack, pen.params_.attack, atol=1e-4)
    assert np.allclose(base.params_.defence, pen.params_.defence, atol=1e-4)
    assert base.params_.home_advantage == pytest.approx(pen.params_.home_advantage, abs=1e-4)
    assert base.params_.rho == pytest.approx(pen.params_.rho, abs=1e-4)
