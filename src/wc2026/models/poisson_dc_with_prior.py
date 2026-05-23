"""PoissonDC with an L2 prior pulling attack/defence toward Elo-derived centres.

We extend :class:`wc2026.models.poisson_dc.PoissonDC` by overriding ``fit`` to
add a quadratic penalty to the negative log-likelihood:

    penalty(theta) = prior_strength * (
        sum_t (attack[t]  - attack_centre[t]) ** 2
      + sum_t (defence[t] - defence_centre[t]) ** 2
    )

The gradient is straightforward — for each team ``t`` we add
``2 * prior_strength * (attack[t] - attack_centre[t])`` to the attack gradient
(and similarly for defence). The sum-to-zero reparameterisation already used
by :func:`_neg_log_lik_and_grad` then folds this through the same
``grad[t] - grad[n-1]`` reduction.

We deliberately do not touch the private base-class function — we re-import it
and add the prior contribution on top. ``prior_strength=0`` reproduces the
unpenalised MLE bit-for-bit.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from wc2026.models.poisson_dc import (
    DEFAULT_MAX_GOALS,
    PoissonDC,
    PoissonDCParams,
    _neg_log_lik_and_grad,
    _unpack_theta,
)


@dataclass(frozen=True)
class _FitInputs:
    """Preprocessed match arrays + team list, ready for the optimizer closure."""

    teams: list[str]
    h_idx: np.ndarray
    a_idx: np.ndarray
    h_score: np.ndarray
    a_score: np.ndarray
    not_neutral: np.ndarray
    w: np.ndarray

    @property
    def n_teams(self) -> int:
        return len(self.teams)


def _prepare_fit_inputs(
    matches: pd.DataFrame, weights: pd.Series | np.ndarray | None
) -> _FitInputs:
    """Validate ``matches``/``weights`` and pack into numpy arrays for fit().

    Mirrors the data-prep half of :meth:`PoissonDC.fit`. Extracted here so the
    wrapper's fit() does not have to duplicate the logic verbatim. (We cannot
    refactor the base class itself per the Stage-1 boundary rules.)
    """
    required = ("home_team", "away_team", "home_score", "away_score", "neutral")
    missing = [c for c in required if c not in matches.columns]
    if missing:
        raise ValueError(f"matches is missing required columns: {missing}")
    if matches.empty:
        raise ValueError("matches is empty")

    teams = sorted(set(matches["home_team"]).union(matches["away_team"]))
    if len(teams) < 2:
        raise ValueError(f"need >= 2 distinct teams, got {len(teams)}")
    team_idx = {t: i for i, t in enumerate(teams)}

    h_idx = matches["home_team"].map(team_idx).to_numpy(dtype=np.int64)
    a_idx = matches["away_team"].map(team_idx).to_numpy(dtype=np.int64)
    h_score = matches["home_score"].to_numpy(dtype=int)
    a_score = matches["away_score"].to_numpy(dtype=int)
    not_neutral = (~matches["neutral"].astype(bool).to_numpy()).astype(float)
    w = np.ones(len(matches), dtype=float) if weights is None else np.asarray(weights, float)
    if w.shape != (len(matches),):
        raise ValueError(f"weights shape {w.shape} != ({len(matches)},)")

    return _FitInputs(
        teams=teams,
        h_idx=h_idx,
        a_idx=a_idx,
        h_score=h_score,
        a_score=a_score,
        not_neutral=not_neutral,
        w=w,
    )


def _penalised_objective(
    theta: np.ndarray,
    *,
    h_idx: np.ndarray,
    a_idx: np.ndarray,
    h_score: np.ndarray,
    a_score: np.ndarray,
    not_neutral: np.ndarray,
    w: np.ndarray,
    n_teams: int,
    attack_centres: np.ndarray,
    defence_centres: np.ndarray,
    prior_strength: float,
) -> tuple[float, np.ndarray]:
    """Base NLL + L2 prior penalty (value, gradient) in the free-param vector."""
    nll, grad = _neg_log_lik_and_grad(
        theta,
        h_idx=h_idx,
        a_idx=a_idx,
        h_score=h_score,
        a_score=a_score,
        not_neutral=not_neutral,
        w=w,
        n_teams=n_teams,
    )
    if prior_strength == 0.0:
        return nll, grad

    atk, dfc, _ha, _rho = _unpack_theta(theta, n_teams)
    atk_diff = atk - attack_centres
    dfc_diff = dfc - defence_centres
    penalty = float(prior_strength * (np.sum(atk_diff**2) + np.sum(dfc_diff**2)))

    pen_grad_atk = 2.0 * prior_strength * atk_diff
    pen_grad_dfc = 2.0 * prior_strength * dfc_diff
    # Sum-to-zero reduction (mirrors _neg_log_lik_and_grad).
    pen_grad_atk_free = pen_grad_atk[: n_teams - 1] - pen_grad_atk[n_teams - 1]
    pen_grad_dfc_free = pen_grad_dfc[: n_teams - 1] - pen_grad_dfc[n_teams - 1]

    pen_grad = np.concatenate([pen_grad_atk_free, pen_grad_dfc_free, [0.0, 0.0]])
    return nll + penalty, grad + pen_grad


class PoissonDCWithPrior(PoissonDC):
    """PoissonDC with an L2 Elo prior on attack/defence parameters.

    Parameters
    ----------
    prior_strength :
        Multiplier on the L2 penalty. ``0`` reduces to the unpenalised
        PoissonDC; values around 0.5-2.0 are typical. The penalty competes
        against the (weighted) data log-likelihood, so the right scale depends
        on the effective sample size.
    """

    def __init__(
        self,
        *,
        attack_centres: np.ndarray | Sequence[float],
        defence_centres: np.ndarray | Sequence[float],
        teams: Sequence[str],
        prior_strength: float = 1.0,
        max_goals: int | None = None,
    ) -> None:
        super().__init__(max_goals=DEFAULT_MAX_GOALS if max_goals is None else max_goals)
        if prior_strength < 0:
            raise ValueError(f"prior_strength must be >= 0, got {prior_strength}")
        teams_t = tuple(teams)
        atk = np.asarray(attack_centres, dtype=float)
        dfc = np.asarray(defence_centres, dtype=float)
        if atk.shape != (len(teams_t),) or dfc.shape != (len(teams_t),):
            raise ValueError(f"centre shapes {atk.shape}/{dfc.shape} != ({len(teams_t)},)")
        # Internally store as a dict so the alignment helper is a simple lookup
        # and the instance state survives team-set changes between fits.
        self._prior_map: dict[str, tuple[float, float]] = {
            t: (float(atk[i]), float(dfc[i])) for i, t in enumerate(teams_t)
        }
        self.prior_strength: float = float(prior_strength)

    def _aligned_centres(self, fit_teams: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Reorder the stored prior to match the team order used inside fit().

        Teams present in the fit but missing from the supplied prior get 0.0.
        """
        n = len(fit_teams)
        atk = np.zeros(n, dtype=float)
        dfc = np.zeros(n, dtype=float)
        for i, t in enumerate(fit_teams):
            pair = self._prior_map.get(t)
            if pair is not None:
                atk[i], dfc[i] = pair
        return atk, dfc

    def fit(
        self,
        matches: pd.DataFrame,
        *,
        weights: pd.Series | np.ndarray | None = None,
        rho_bounds: tuple[float, float] = (-0.2, 0.2),
        home_advantage_bounds: tuple[float, float] = (-0.5, 1.5),
        max_iter: int = 500,
        tol: float = 1e-6,
    ) -> PoissonDCWithPrior:
        inputs = _prepare_fit_inputs(matches, weights)
        self._team_idx = {t: i for i, t in enumerate(inputs.teams)}
        attack_centres, defence_centres = self._aligned_centres(inputs.teams)

        n = inputs.n_teams
        n_free = 2 * (n - 1) + 2
        x0 = np.zeros(n_free)
        x0[2 * (n - 1)] = 0.3  # plausible home advantage prior
        bounds: list[tuple[float | None, float | None]] = [(None, None)] * (2 * (n - 1)) + [
            home_advantage_bounds,
            rho_bounds,
        ]

        def closure(theta: np.ndarray) -> tuple[float, np.ndarray]:
            return _penalised_objective(
                theta,
                h_idx=inputs.h_idx,
                a_idx=inputs.a_idx,
                h_score=inputs.h_score,
                a_score=inputs.a_score,
                not_neutral=inputs.not_neutral,
                w=inputs.w,
                n_teams=n,
                attack_centres=attack_centres,
                defence_centres=defence_centres,
                prior_strength=self.prior_strength,
            )

        result = minimize(
            closure,
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
        )
        if not result.success:
            warnings.warn(
                f"PoissonDCWithPrior.fit did not converge: {result.message!r} "
                f"(n_iter={result.nit}, nll={result.fun:.4f})",
                RuntimeWarning,
                stacklevel=2,
            )
        atk, dfc, ha, rho = _unpack_theta(result.x, n)
        self.params_ = PoissonDCParams(
            teams=tuple(inputs.teams), attack=atk, defence=dfc, home_advantage=ha, rho=rho
        )
        self.converged_ = bool(result.success)
        self.n_iter_ = int(result.nit)
        self.final_nll_ = float(result.fun)
        return self
