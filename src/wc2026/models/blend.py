"""Geometric-mean blend of two 1X2 probability distributions.

The blueprint (Phase 5) calls for a hybrid of the structural Poisson-DC model
and the XGBoost ML classifier. We use the **geometric mean** of the two
probability triplets and renormalise — this is the same ensemble convention
used by Groll, Ley, Schauberger & Van Eetvelde (2019, *JQAS* 15(4)):

    p_blend[k] ∝ p_poisson[k] ** w_p · p_xgb[k] ** w_x

with ``w_p + w_x == 1``. The default 50/50 split is documented; callers can
tilt either way by passing a different ``weight``.

Why geometric not arithmetic
----------------------------
Arithmetic mean of two well-calibrated distributions is also well-calibrated
in expectation, but geometric mean preserves the "log-odds" structure that
makes both inputs interpretable — a 70% home prediction from each model
combines to 70% home, not the arithmetic 70%-and-shifted-by-margin. It also
matches the convention used in the published academic literature.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

EPS = 1e-12


def _to_array(p: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.shape[-1] != 3:
        raise ValueError(f"blend expects shape (n, 3) probabilities; got {arr.shape}")
    return arr


def blend_geometric(
    p_poisson: Sequence[float] | np.ndarray,
    p_xgb: Sequence[float] | np.ndarray,
    *,
    weight: float = 0.5,
) -> np.ndarray:
    """Geometric blend of two ``(n, 3)`` (or ``(3,)``) probability arrays.

    ``weight`` is the *Poisson* mixing weight in [0, 1]; XGBoost gets ``1 - weight``.
    Both inputs are clipped to ``[EPS, 1]`` before exponentiation so a model
    that emits a clean 0 doesn't NaN the blend. The result is renormalised so
    every row sums to 1.

    Returns ``(n, 3)`` (or ``(3,)`` if both inputs were ``(3,)``).
    """
    if not (0.0 <= weight <= 1.0):
        raise ValueError(f"weight must be in [0, 1]; got {weight}")
    poisson_arr = _to_array(p_poisson)
    xgb_arr = _to_array(p_xgb)
    if poisson_arr.shape != xgb_arr.shape:
        raise ValueError(
            f"blend operands must have the same shape; got {poisson_arr.shape} vs {xgb_arr.shape}"
        )
    poisson_arr = np.clip(poisson_arr, EPS, 1.0)
    xgb_arr = np.clip(xgb_arr, EPS, 1.0)
    blended = poisson_arr**weight * xgb_arr ** (1.0 - weight)
    row_sums = blended.sum(axis=-1, keepdims=True)
    out = blended / np.where(row_sums == 0.0, 1.0, row_sums)
    # Squeeze back to (3,) when both inputs were scalar.
    if np.asarray(p_poisson, dtype=float).ndim == 1 and np.asarray(p_xgb, dtype=float).ndim == 1:
        return out[0]
    return out


def blend_dict(
    poisson: dict[str, float],
    xgb: dict[str, float],
    *,
    weight: float = 0.5,
) -> dict[str, float]:
    """Convenience wrapper for the ``{home_win, draw, away_win}`` dict shape.

    Used directly by ``api.routes.predictions`` so the route doesn't have to
    juggle numpy arrays.
    """
    keys = ("home_win", "draw", "away_win")
    p_p = np.array([poisson[k] for k in keys], dtype=float)
    p_x = np.array([xgb[k] for k in keys], dtype=float)
    blended = blend_geometric(p_p, p_x, weight=weight)
    return {k: float(v) for k, v in zip(keys, blended, strict=True)}


__all__ = ["EPS", "blend_dict", "blend_geometric"]
