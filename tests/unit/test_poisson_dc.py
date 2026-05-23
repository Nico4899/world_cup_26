"""Unit tests for the Poisson + Dixon-Coles model."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from wc2026.models.poisson_dc import (
    PoissonDC,
    PoissonDCParams,
    _neg_log_lik_and_grad,
    dixon_coles_tau,
    log_poisson_pmf,
)

# --- pure-function tests ----------------------------------------------------


def test_tau_corner_cases_with_known_values() -> None:
    lh, la, rho = 1.4, 1.1, 0.05
    # 0-0
    assert math.isclose(dixon_coles_tau(0, 0, lh, la, rho), 1.0 - lh * la * rho)
    # 0-1
    assert math.isclose(dixon_coles_tau(0, 1, lh, la, rho), 1.0 + lh * rho)
    # 1-0
    assert math.isclose(dixon_coles_tau(1, 0, lh, la, rho), 1.0 + la * rho)
    # 1-1
    assert math.isclose(dixon_coles_tau(1, 1, lh, la, rho), 1.0 - rho)
    # non-corner cells unchanged
    for i, j in [(2, 0), (0, 2), (2, 2), (3, 1), (1, 3)]:
        assert dixon_coles_tau(i, j, lh, la, rho) == 1.0


def test_tau_when_rho_zero_returns_one_everywhere() -> None:
    h = np.array([0, 0, 1, 1, 2, 2])
    a = np.array([0, 1, 0, 1, 0, 2])
    lh = np.full(6, 1.5)
    la = np.full(6, 1.2)
    tau = dixon_coles_tau(h, a, lh, la, rho=0.0)
    assert np.allclose(tau, 1.0)


def test_tau_vectorised_matches_scalar() -> None:
    rng = np.random.default_rng(0)
    h = rng.integers(0, 5, 20)
    a = rng.integers(0, 5, 20)
    lh = rng.uniform(0.3, 3.0, 20)
    la = rng.uniform(0.3, 3.0, 20)
    rho = 0.07
    vec = dixon_coles_tau(h, a, lh, la, rho)
    for i, (hi, ai, lhi, lai) in enumerate(zip(h, a, lh, la, strict=True)):
        assert math.isclose(
            vec[i], float(dixon_coles_tau(int(hi), int(ai), float(lhi), float(lai), rho))
        )


def test_analytic_gradient_matches_finite_difference() -> None:
    """At a random non-trivial theta, analytic gradient must match scipy's finite
    difference approximation. Catches sign errors and missing chain-rule factors."""
    from scipy.optimize import approx_fprime

    rng = np.random.default_rng(123)
    n_teams = 6
    n_matches = 400
    h_idx = rng.integers(0, n_teams, n_matches).astype(np.int64)
    a_idx = rng.integers(0, n_teams, n_matches).astype(np.int64)
    same = h_idx == a_idx
    a_idx[same] = (a_idx[same] + 1) % n_teams
    h_score = rng.integers(0, 5, n_matches).astype(int)
    a_score = rng.integers(0, 5, n_matches).astype(int)
    not_neutral = (rng.random(n_matches) > 0.3).astype(float)
    w = rng.uniform(0.1, 2.0, n_matches)

    # random theta with non-zero rho and non-zero home advantage so all gradient
    # paths (including all four DC tau corners) are exercised.
    n_free = 2 * (n_teams - 1) + 2
    theta = rng.normal(0.0, 0.2, n_free)
    theta[2 * (n_teams - 1)] = 0.4  # home advantage
    theta[2 * (n_teams - 1) + 1] = 0.05  # rho

    kwargs = dict(
        h_idx=h_idx,
        a_idx=a_idx,
        h_score=h_score,
        a_score=a_score,
        not_neutral=not_neutral,
        w=w,
        n_teams=n_teams,
    )
    _, grad_analytic = _neg_log_lik_and_grad(theta, **kwargs)

    def f_only(t: np.ndarray) -> float:
        return _neg_log_lik_and_grad(t, **kwargs)[0]

    grad_numeric = approx_fprime(theta, f_only, epsilon=1e-6)
    # finite-difference accuracy ~ epsilon * |f| ≈ 1e-6 * a few thousand ≈ a few mille;
    # rel error 1e-3 is generous, abs 1e-2 covers near-zero components.
    rel_err = np.abs(grad_analytic - grad_numeric) / (np.abs(grad_numeric) + 1e-6)
    assert rel_err.max() < 5e-3, (
        f"max rel err {rel_err.max():.2e}; analytic={grad_analytic[:5]}; numeric={grad_numeric[:5]}"
    )


def test_log_poisson_pmf_matches_scipy() -> None:
    rng = np.random.default_rng(1)
    ks = rng.integers(0, 8, 30)
    lams = rng.uniform(0.2, 4.0, 30)
    ours = log_poisson_pmf(ks, lams)
    theirs = poisson.logpmf(ks, lams)
    assert np.allclose(ours, theirs, atol=1e-10)


# --- container --------------------------------------------------------------


def test_params_post_init_shape_check() -> None:
    with pytest.raises(ValueError, match="attack shape"):
        PoissonDCParams(
            teams=("A", "B"),
            attack=np.zeros(3),
            defence=np.zeros(2),
            home_advantage=0.3,
            rho=0.0,
        )


# --- inference tests (using a hand-set params object) ----------------------


def _make_seeded_model(rho: float = 0.0) -> PoissonDC:
    """Return a model with hand-set params: two teams 'A' (strong) and 'B' (weak)."""
    m = PoissonDC(max_goals=8)
    m._team_idx = {"A": 0, "B": 1}
    m.params_ = PoissonDCParams(
        teams=("A", "B"),
        attack=np.array([0.25, -0.25]),
        defence=np.array([-0.20, 0.20]),
        home_advantage=0.30,
        rho=rho,
    )
    m.converged_ = True
    return m


def test_score_probs_sums_to_one() -> None:
    m = _make_seeded_model(rho=0.07)
    p = m.score_probs("A", "B")
    assert math.isclose(p.sum(), 1.0, abs_tol=1e-12)


def test_outcome_probs_sum_to_one() -> None:
    m = _make_seeded_model(rho=0.07)
    out = m.outcome_probs("A", "B")
    assert math.isclose(out["home_win"] + out["draw"] + out["away_win"], 1.0, abs_tol=1e-12)


def test_score_probs_marginals_match_poisson_when_rho_zero() -> None:
    m = _make_seeded_model(rho=0.0)
    p = m.score_probs("A", "B")
    lh, la = m.expected_goals("A", "B")
    # row marginals → P(home_goals = i) up to truncation
    row_marg = p.sum(axis=1)
    expected_h = poisson.pmf(np.arange(m.max_goals + 1), lh)
    expected_h = expected_h / expected_h.sum()  # normalise the truncation
    assert np.allclose(row_marg, expected_h, atol=1e-8)
    # column marginals → P(away_goals = j)
    col_marg = p.sum(axis=0)
    expected_a = poisson.pmf(np.arange(m.max_goals + 1), la)
    expected_a = expected_a / expected_a.sum()
    assert np.allclose(col_marg, expected_a, atol=1e-8)


def test_home_advantage_increases_home_win_probability() -> None:
    m = _make_seeded_model(rho=0.05)
    home_p = m.outcome_probs("A", "B", neutral=False)
    neutral_p = m.outcome_probs("A", "B", neutral=True)
    # at neutral, A loses its home boost → home_win should drop, away_win should rise
    assert neutral_p["home_win"] < home_p["home_win"]
    assert neutral_p["away_win"] > home_p["away_win"]


def test_stronger_attack_wins_more() -> None:
    m = _make_seeded_model(rho=0.05)
    # A is strong (attack +0.25, defence -0.20) → A > B at home
    a_at_home = m.outcome_probs("A", "B", neutral=False)
    b_at_home = m.outcome_probs("B", "A", neutral=False)
    assert a_at_home["home_win"] > a_at_home["away_win"]
    # at A's home, A win > 0.5; at B's home, B win is moderately probable but < A's prob at home
    assert b_at_home["home_win"] < a_at_home["home_win"]


def test_expected_goals_unknown_team_raises() -> None:
    m = _make_seeded_model()
    with pytest.raises(KeyError, match="unknown team"):
        m.expected_goals("A", "ZZ")


def test_inference_before_fit_raises() -> None:
    m = PoissonDC()
    with pytest.raises(RuntimeError, match="not fitted"):
        m.expected_goals("A", "B")


# --- fit tests --------------------------------------------------------------


def test_fit_requires_columns() -> None:
    df = pd.DataFrame({"home_team": ["A"], "away_team": ["B"]})
    with pytest.raises(ValueError, match="missing required columns"):
        PoissonDC().fit(df)


def test_fit_rejects_empty() -> None:
    df = pd.DataFrame(
        {
            "home_team": [],
            "away_team": [],
            "home_score": [],
            "away_score": [],
            "neutral": [],
        }
    )
    with pytest.raises(ValueError, match="empty"):
        PoissonDC().fit(df)


def test_fit_rejects_single_team() -> None:
    df = pd.DataFrame(
        {
            "home_team": ["A"],
            "away_team": ["A"],
            "home_score": [1],
            "away_score": [0],
            "neutral": [False],
        }
    )
    with pytest.raises(ValueError, match=">= 2 distinct teams"):
        PoissonDC().fit(df)


def _simulate_matches(
    teams: list[str],
    attack: np.ndarray,
    defence: np.ndarray,
    home_advantage: float,
    rho: float,
    n_matches: int,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(teams)
    h_idx = rng.integers(0, n, n_matches)
    a_idx = rng.integers(0, n, n_matches)
    # avoid self-matches
    same = h_idx == a_idx
    a_idx[same] = (a_idx[same] + 1) % n
    neutral = rng.random(n_matches) < 0.2
    lh = np.exp(attack[h_idx] + defence[a_idx] + home_advantage * (~neutral).astype(float))
    la = np.exp(attack[a_idx] + defence[h_idx])
    # naive sampler: ignores DC tau for the simulation (small rho ⇒ tiny bias)
    h_score = rng.poisson(lh)
    a_score = rng.poisson(la)
    _ = rho  # documented unused; the simulator is independent-Poisson on purpose
    return pd.DataFrame(
        {
            "home_team": [teams[i] for i in h_idx],
            "away_team": [teams[i] for i in a_idx],
            "home_score": h_score,
            "away_score": a_score,
            "neutral": neutral,
        }
    )


def test_fit_recovers_known_attack_defence_on_simulated_data() -> None:
    """Generate 8k matches from known params and confirm the fit recovers them."""
    teams = list("ABCDEFGH")
    true_attack = np.array([0.40, 0.25, 0.10, 0.00, -0.05, -0.15, -0.25, -0.30])
    true_defence = np.array([-0.30, -0.20, -0.10, 0.00, 0.05, 0.15, 0.20, 0.20])
    true_ha = 0.35
    matches = _simulate_matches(teams, true_attack, true_defence, true_ha, rho=0.0, n_matches=8000)
    model = PoissonDC().fit(matches)
    assert model.converged_
    # team ordering is alphabetical; matches our 'teams' list
    assert model.params_.teams == tuple(teams)
    # attack and defence should be close to the truth (within 0.10 in log-rate)
    assert np.allclose(model.params_.attack, true_attack, atol=0.10)
    assert np.allclose(model.params_.defence, true_defence, atol=0.10)
    assert abs(model.params_.home_advantage - true_ha) < 0.10
    # sum-to-zero constraint is exact
    assert abs(model.params_.attack.sum()) < 1e-8
    assert abs(model.params_.defence.sum()) < 1e-8


def test_fit_uses_weights() -> None:
    """Two simulated datasets stitched together: 'good' rows from one truth and
    'noise' rows from a different truth. With weights zeroing out the noise, the
    fit should recover the good truth (not the marginal average of the two).
    """
    teams = list("ABCD")
    true_attack = np.array([0.40, 0.10, -0.20, -0.30])
    true_defence = np.array([-0.30, -0.10, 0.15, 0.25])
    true_ha = 0.30
    good = _simulate_matches(
        teams, true_attack, true_defence, true_ha, rho=0.0, n_matches=4000, seed=11
    )
    # noise: same teams, but very different "truth" (uniform, no signal)
    noise_attack = np.zeros(4)
    noise_defence = np.zeros(4)
    noise = _simulate_matches(
        teams, noise_attack, noise_defence, 0.0, rho=0.0, n_matches=4000, seed=22
    )
    matches = pd.concat([good, noise], ignore_index=True)
    weights = np.concatenate([np.ones(len(good)), np.zeros(len(noise))])

    model = PoissonDC().fit(matches, weights=weights)
    assert model.converged_
    # weighted fit should recover the 'good' truth, not the average
    assert np.allclose(model.params_.attack, true_attack, atol=0.12)
    assert np.allclose(model.params_.defence, true_defence, atol=0.12)
    assert abs(model.params_.home_advantage - true_ha) < 0.12
