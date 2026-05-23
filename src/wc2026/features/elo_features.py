"""Elo-ratings features used to centre the PoissonDC attack/defence priors.

The eloratings.net snapshot gives a single scalar ``rating`` per team. We turn
that into a (attack, defence) prior pair under the convention

    strength       = (rating - mean(rating)) / 100
    attack_prior   = +strength
    defence_prior  = -strength

The intuition: a +100-Elo team should, all else equal, score more and concede
less. Centring at the snapshot mean preserves the PoissonDC sum-to-zero
identifiability. The /100 scaling is a soft tunable; the actual influence is
controlled by ``prior_strength`` in :class:`PoissonDCWithPrior`.

Reference
---------
World Football Elo Ratings, https://www.eloratings.net (snapshot loaded via
``wc2026.ingest.eloratings_scraper.load_latest_snapshot``).
"""

from __future__ import annotations

import pandas as pd

ELO_SCALE: float = 100.0


def elo_to_strength_prior(elo_snapshot_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Map ``team_name -> (attack_prior, defence_prior)`` from an Elo snapshot.

    Parameters
    ----------
    elo_snapshot_df :
        DataFrame with ``team_name`` and ``rating`` columns (as returned by
        :func:`wc2026.ingest.eloratings_scraper.load_latest_snapshot`).

    Returns
    -------
    Dict keyed by ``team_name``. Each value is a ``(attack_prior, defence_prior)``
    tuple with ``defence_prior = -attack_prior``.
    """
    required = {"team_name", "rating"}
    missing = required - set(elo_snapshot_df.columns)
    if missing:
        raise ValueError(f"elo snapshot missing columns: {sorted(missing)}")
    if elo_snapshot_df.empty:
        return {}
    ratings = elo_snapshot_df["rating"].astype(float)
    mean_elo = float(ratings.mean())
    out: dict[str, tuple[float, float]] = {}
    for name, rating in zip(elo_snapshot_df["team_name"], ratings, strict=True):
        strength = (float(rating) - mean_elo) / ELO_SCALE
        out[str(name)] = (strength, -strength)
    return out
