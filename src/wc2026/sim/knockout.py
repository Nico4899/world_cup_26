"""Knockout-stage single-match simulator.

Each knockout fixture proceeds through up to three phases:

    1. Regulation (90 min): scoreline sampled from ``model.score_probs``.
    2. Extra time   (30 min): if regulation is tied, each side draws an integer
       number of additional goals from Poisson(lambda_90 / 3). Rationale: ET is
       30 minutes = 1/3 of regulation, so the same per-minute scoring rate gives
       1/3 of the expected goal count.
    3. Penalty shootout: if still tied after ET, **50/50 coin flip** as a
       placeholder. This is a known weakness flagged in the review plan — a
       proper shootout submodel (Stage 0.6 candidate) would condition on Elo
       differential and historical shootout win rates.

Knockout matches are always treated as ``neutral=True`` (FIFA neutral-venue rule
for the knockout stage), even though the venue happens to be in the USA / Mexico /
Canada. The model's home_advantage is therefore not applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from wc2026.models.poisson_dc import PoissonDC
from wc2026.sim.groups import sample_scoreline

DecidedIn = Literal["regulation", "extra_time", "shootout"]


@dataclass(frozen=True)
class KnockoutOutcome:
    home_team: str
    away_team: str
    regulation_score: tuple[int, int]            # (h, a) after 90 minutes
    extra_time_score: tuple[int, int] | None     # additional goals in ET; None if not played
    shootout_winner: str | None                  # team name; None if not played
    winner: str
    decided_in: DecidedIn

    @property
    def total_score(self) -> tuple[int, int]:
        """Aggregate (h, a) including ET goals (excludes shootout outcome)."""
        if self.extra_time_score is None:
            return self.regulation_score
        return (
            self.regulation_score[0] + self.extra_time_score[0],
            self.regulation_score[1] + self.extra_time_score[1],
        )


def simulate_knockout_match(
    home_team: str,
    away_team: str,
    model: PoissonDC,
    rng: np.random.Generator,
    *,
    neutral: bool = True,
) -> KnockoutOutcome:
    """Simulate one knockout match end-to-end. See module docstring for phases."""
    prob = model.score_probs(home_team, away_team, neutral=neutral)
    h, a = sample_scoreline(prob, rng)
    reg = (h, a)

    if h != a:
        winner = home_team if h > a else away_team
        return KnockoutOutcome(
            home_team=home_team,
            away_team=away_team,
            regulation_score=reg,
            extra_time_score=None,
            shootout_winner=None,
            winner=winner,
            decided_in="regulation",
        )

    # Tied → extra time. Use the model's expected goals to set ET rates at 1/3.
    lh90, la90 = model.expected_goals(home_team, away_team, neutral=neutral)
    et_h = int(rng.poisson(lh90 / 3.0))
    et_a = int(rng.poisson(la90 / 3.0))
    et = (et_h, et_a)

    if et_h != et_a:
        winner = home_team if et_h > et_a else away_team
        return KnockoutOutcome(
            home_team=home_team,
            away_team=away_team,
            regulation_score=reg,
            extra_time_score=et,
            shootout_winner=None,
            winner=winner,
            decided_in="extra_time",
        )

    # Still tied after ET → shootout (50/50 placeholder).
    shootout_winner = home_team if rng.random() < 0.5 else away_team
    return KnockoutOutcome(
        home_team=home_team,
        away_team=away_team,
        regulation_score=reg,
        extra_time_score=et,
        shootout_winner=shootout_winner,
        winner=shootout_winner,
        decided_in="shootout",
    )
