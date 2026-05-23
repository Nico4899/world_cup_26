"""Map an Elo snapshot to per-team prior centres for PoissonDCWithPrior.

Given the list of teams used by a fitted :class:`PoissonDC` (in the same order
as ``PoissonDCParams.teams``) and an Elo snapshot, return two numpy arrays of
length ``n_teams`` aligned to that order:

    attack_centres[i], defence_centres[i]

Teams missing from the Elo snapshot fall back to ``(0.0, 0.0)``. That makes the
prior a no-op for unknown teams instead of dragging them toward the snapshot
mean, which is the safer default.

This module deliberately lives separately from
:mod:`wc2026.features.elo_features` so that the per-team mapping (which depends
only on the snapshot) is reusable outside model fitting.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from wc2026.features.elo_features import elo_to_strength_prior


def elo_prior_centres(
    teams: Sequence[str], elo_snapshot_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(attack_centres, defence_centres)`` aligned to ``teams``.

    Missing teams (not in the Elo snapshot) get ``0.0`` for both centres.
    """
    prior_map = elo_to_strength_prior(elo_snapshot_df)
    n = len(teams)
    attack_centres = np.zeros(n, dtype=float)
    defence_centres = np.zeros(n, dtype=float)
    for i, team in enumerate(teams):
        if team in prior_map:
            a, d = prior_map[team]
            attack_centres[i] = a
            defence_centres[i] = d
    return attack_centres, defence_centres
