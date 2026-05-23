"""Independent Poisson + Dixon-Coles model for international football matches.

For each match the expected goal counts are

    lambda_home = exp(attack[home] + defence[away] + home_advantage * (1 - neutral))
    lambda_away = exp(attack[away] + defence[home])

Goals are conditionally Poisson, with the Dixon-Coles low-score correction tau
that adjusts only the four corner cells (0-0, 0-1, 1-0, 1-1):

    tau(0, 0) = 1 - lambda_home * lambda_away * rho
    tau(0, 1) = 1 + lambda_home * rho
    tau(1, 0) = 1 + lambda_away * rho
    tau(1, 1) = 1 - rho
    tau(i, j) = 1 otherwise

Identifiability is fixed by enforcing sum(attack) = 0 and sum(defence) = 0
across all teams. The optimizer therefore sees 2*(N-1) free strength parameters
plus the home-advantage scalar and rho.

Reference
---------
Dixon & Coles (1997), "Modelling Association Football Scores and Inefficiencies
in the Football Betting Market", JRSS Series C (Applied Statistics) 46(2):265-280.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

DEFAULT_MAX_GOALS = 10
"""Truncation cap for the score matrix; covers >99.99% of realistic mass."""


# --- pure functions ---------------------------------------------------------


def dixon_coles_tau(
    home_goals: np.ndarray | int,
    away_goals: np.ndarray | int,
    lambda_home: np.ndarray | float,
    lambda_away: np.ndarray | float,
    rho: float,
) -> np.ndarray:
    """Dixon-Coles low-score correction. Vectorised over goals + rates."""
    h = np.asarray(home_goals, dtype=int)
    a = np.asarray(away_goals, dtype=int)
    lh = np.asarray(lambda_home, dtype=float)
    la = np.asarray(lambda_away, dtype=float)
    tau = np.ones(np.broadcast_shapes(h.shape, a.shape, lh.shape, la.shape), dtype=float)
    tau = np.where((h == 0) & (a == 0), 1.0 - lh * la * rho, tau)
    tau = np.where((h == 0) & (a == 1), 1.0 + lh * rho, tau)
    tau = np.where((h == 1) & (a == 0), 1.0 + la * rho, tau)
    tau = np.where((h == 1) & (a == 1), 1.0 - rho, tau)
    return tau


def log_poisson_pmf(k: np.ndarray | int, lam: np.ndarray | float) -> np.ndarray:
    """log P(K = k) under Poisson(lam). Vectorised."""
    k_arr = np.asarray(k, dtype=float)
    lam_arr = np.asarray(lam, dtype=float)
    return -lam_arr + k_arr * np.log(lam_arr) - gammaln(k_arr + 1.0)


# --- parameter container ----------------------------------------------------


@dataclass(frozen=True)
class PoissonDCParams:
    teams: tuple[str, ...]
    attack: np.ndarray
    defence: np.ndarray
    home_advantage: float
    rho: float

    def __post_init__(self) -> None:
        if self.attack.shape != (len(self.teams),):
            raise ValueError(f"attack shape {self.attack.shape}, expected ({len(self.teams)},)")
        if self.defence.shape != (len(self.teams),):
            raise ValueError(f"defence shape {self.defence.shape}, expected ({len(self.teams)},)")


# --- internal: parameter packing + log-likelihood with analytic gradient ----


def _unpack_theta(theta: np.ndarray, n_teams: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Unpack the flat free-param vector into (attack, defence, home_adv, rho).

    The first 2*(n-1) entries are free attack/defence params; the last team's
    attack and defence are derived from sum-to-zero. Last two entries are
    home_advantage and rho.
    """
    atk_free = theta[: n_teams - 1]
    def_free = theta[n_teams - 1 : 2 * (n_teams - 1)]
    ha = float(theta[2 * (n_teams - 1)])
    rho = float(theta[2 * (n_teams - 1) + 1])
    atk = np.empty(n_teams)
    atk[: n_teams - 1] = atk_free
    atk[n_teams - 1] = -atk_free.sum()
    dfc = np.empty(n_teams)
    dfc[: n_teams - 1] = def_free
    dfc[n_teams - 1] = -def_free.sum()
    return atk, dfc, ha, rho


def _neg_log_lik_and_grad(
    theta: np.ndarray,
    *,
    h_idx: np.ndarray,
    a_idx: np.ndarray,
    h_score: np.ndarray,
    a_score: np.ndarray,
    not_neutral: np.ndarray,
    w: np.ndarray,
    n_teams: int,
) -> tuple[float, np.ndarray]:
    """Compute the weighted negative log-likelihood and its analytic gradient.

    Used by `PoissonDC.fit`; exposed for unit testing (compared against a
    finite-difference gradient).
    """
    atk, dfc, ha, rho = _unpack_theta(theta, n_teams)
    lh = np.exp(atk[h_idx] + dfc[a_idx] + ha * not_neutral)
    la = np.exp(atk[a_idx] + dfc[h_idx])
    tau_raw = dixon_coles_tau(h_score, a_score, lh, la, rho)
    tau = np.maximum(tau_raw, 1e-10)
    ll = log_poisson_pmf(h_score, lh) + log_poisson_pmf(a_score, la) + np.log(tau)
    nll = float(-np.sum(w * ll))

    # d log(tau) / d lh, d la, d rho — only nonzero on the four corner cells.
    d_lt_d_lh = np.zeros_like(lh)
    d_lt_d_la = np.zeros_like(la)
    d_lt_d_rho = np.zeros_like(lh)
    m00 = (h_score == 0) & (a_score == 0)
    m01 = (h_score == 0) & (a_score == 1)
    m10 = (h_score == 1) & (a_score == 0)
    m11 = (h_score == 1) & (a_score == 1)
    if m00.any():
        d_lt_d_lh[m00] = -la[m00] * rho / tau[m00]
        d_lt_d_la[m00] = -lh[m00] * rho / tau[m00]
        d_lt_d_rho[m00] = -lh[m00] * la[m00] / tau[m00]
    if m01.any():
        d_lt_d_lh[m01] = rho / tau[m01]
        d_lt_d_rho[m01] = lh[m01] / tau[m01]
    if m10.any():
        d_lt_d_la[m10] = rho / tau[m10]
        d_lt_d_rho[m10] = la[m10] / tau[m10]
    if m11.any():
        d_lt_d_rho[m11] = -1.0 / tau[m11]

    # Per-match contributions to d NLL / d lh and d NLL / d la, already
    # multiplied by the chain-rule factor d lh / d (log-rate) = lh.
    grad_lh_pm = w * ((lh - h_score) - lh * d_lt_d_lh)
    grad_la_pm = w * ((la - a_score) - la * d_lt_d_la)
    grad_rho = float(-np.sum(w * d_lt_d_rho))

    # Accumulate per team via bincount (one pass each, O(m)).
    grad_atk = np.bincount(h_idx, grad_lh_pm, minlength=n_teams) + np.bincount(
        a_idx, grad_la_pm, minlength=n_teams
    )
    grad_dfc = np.bincount(a_idx, grad_lh_pm, minlength=n_teams) + np.bincount(
        h_idx, grad_la_pm, minlength=n_teams
    )
    grad_ha = float(np.sum(grad_lh_pm * not_neutral))

    # Sum-to-zero: free params are 0..n-2; the last team is -sum of others, so
    # for each free t the effective gradient is grad[t] - grad[n-1].
    grad_atk_free = grad_atk[: n_teams - 1] - grad_atk[n_teams - 1]
    grad_dfc_free = grad_dfc[: n_teams - 1] - grad_dfc[n_teams - 1]

    grad = np.concatenate([grad_atk_free, grad_dfc_free, [grad_ha, grad_rho]])
    return nll, grad


# --- model ------------------------------------------------------------------


class PoissonDC:
    """Weighted-MLE Independent Poisson + Dixon-Coles model."""

    def __init__(self, *, max_goals: int = DEFAULT_MAX_GOALS) -> None:
        self.max_goals = max_goals
        self.params_: PoissonDCParams | None = None
        self._team_idx: dict[str, int] = {}
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.final_nll_: float | None = None

    @property
    def fitted(self) -> bool:
        return self.params_ is not None

    def fit(
        self,
        matches: pd.DataFrame,
        *,
        weights: pd.Series | np.ndarray | None = None,
        rho_bounds: tuple[float, float] = (-0.2, 0.2),
        home_advantage_bounds: tuple[float, float] = (-0.5, 1.5),
        max_iter: int = 500,
        tol: float = 1e-6,
    ) -> PoissonDC:
        """Fit attack/defence/home_advantage/rho by weighted MLE.

        ``matches`` must have columns:
            home_team, away_team, home_score, away_score, neutral

        ``weights`` is an optional per-row weight (default: all 1s).
        """
        required = ("home_team", "away_team", "home_score", "away_score", "neutral")
        missing = [c for c in required if c not in matches.columns]
        if missing:
            raise ValueError(f"matches is missing required columns: {missing}")
        if matches.empty:
            raise ValueError("matches is empty")

        teams = sorted(set(matches["home_team"]).union(matches["away_team"]))
        n = len(teams)
        if n < 2:
            raise ValueError(f"need >= 2 distinct teams, got {n}")
        self._team_idx = {t: i for i, t in enumerate(teams)}

        h_idx = matches["home_team"].map(self._team_idx).to_numpy(dtype=np.int64)
        a_idx = matches["away_team"].map(self._team_idx).to_numpy(dtype=np.int64)
        h_score = matches["home_score"].to_numpy(dtype=int)
        a_score = matches["away_score"].to_numpy(dtype=int)
        not_neutral = (~matches["neutral"].astype(bool).to_numpy()).astype(float)
        w = np.ones(len(matches), dtype=float) if weights is None else np.asarray(weights, float)
        if w.shape != (len(matches),):
            raise ValueError(f"weights shape {w.shape} != ({len(matches)},)")

        # Pack: [atk_0..atk_{n-2}, def_0..def_{n-2}, home_adv, rho]; last team is sum-to-zero.
        n_free = 2 * (n - 1) + 2
        x0 = np.zeros(n_free)
        x0[2 * (n - 1)] = 0.3  # plausible home advantage prior

        bounds: list[tuple[float | None, float | None]] = [(None, None)] * (2 * (n - 1))
        bounds.append(home_advantage_bounds)
        bounds.append(rho_bounds)

        def closure(theta: np.ndarray) -> tuple[float, np.ndarray]:
            return _neg_log_lik_and_grad(
                theta,
                h_idx=h_idx,
                a_idx=a_idx,
                h_score=h_score,
                a_score=a_score,
                not_neutral=not_neutral,
                w=w,
                n_teams=n,
            )

        result = minimize(
            closure,
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
        )
        atk, dfc, ha, rho = _unpack_theta(result.x, n)
        self.params_ = PoissonDCParams(
            teams=tuple(teams), attack=atk, defence=dfc, home_advantage=ha, rho=rho
        )
        self.converged_ = bool(result.success)
        self.n_iter_ = int(result.nit)
        self.final_nll_ = float(result.fun)
        return self

    # --- inference ----------------------------------------------------------

    def _require_team(self, team: str) -> int:
        if team not in self._team_idx:
            raise KeyError(f"unknown team: {team!r}")
        return self._team_idx[team]

    def _require_fitted(self) -> PoissonDCParams:
        if self.params_ is None:
            raise RuntimeError("model not fitted; call fit() first")
        return self.params_

    def expected_goals(
        self, home_team: str, away_team: str, *, neutral: bool = False
    ) -> tuple[float, float]:
        """Return (lambda_home, lambda_away) for a matchup."""
        p = self._require_fitted()
        h, a = self._require_team(home_team), self._require_team(away_team)
        ha_term = 0.0 if neutral else p.home_advantage
        lh = float(np.exp(p.attack[h] + p.defence[a] + ha_term))
        la = float(np.exp(p.attack[a] + p.defence[h]))
        return lh, la

    def score_probs(self, home_team: str, away_team: str, *, neutral: bool = False) -> np.ndarray:
        """Joint probability matrix of shape (max_goals+1, max_goals+1).

        Rows index home goals (0..max_goals), columns index away goals. Sums to 1 by
        construction (the Poisson tail beyond max_goals is folded into the
        normalisation, which is a sub-0.01% correction at max_goals=10).
        """
        p = self._require_fitted()
        lh, la = self.expected_goals(home_team, away_team, neutral=neutral)
        m = self.max_goals + 1
        ph = poisson.pmf(np.arange(m), lh)
        pa = poisson.pmf(np.arange(m), la)
        prob = np.outer(ph, pa)
        # Apply tau on the four corners.
        prob[0, 0] *= max(1.0 - lh * la * p.rho, 0.0)
        prob[0, 1] *= max(1.0 + lh * p.rho, 0.0)
        prob[1, 0] *= max(1.0 + la * p.rho, 0.0)
        prob[1, 1] *= max(1.0 - p.rho, 0.0)
        total = prob.sum()
        if total <= 0.0:
            raise FloatingPointError("score-probability matrix non-positive total")
        return prob / total

    def outcome_probs(
        self, home_team: str, away_team: str, *, neutral: bool = False
    ) -> dict[str, float]:
        """Return {'home_win': p, 'draw': p, 'away_win': p}."""
        prob = self.score_probs(home_team, away_team, neutral=neutral)
        diag = float(np.trace(prob))
        home_win = float(np.tril(prob, k=-1).sum())
        away_win = float(np.triu(prob, k=1).sum())
        return {"home_win": home_win, "draw": diag, "away_win": away_win}
