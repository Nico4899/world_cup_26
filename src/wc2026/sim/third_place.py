"""Best-third-placed ranking across the 12 groups.

In the 48-team format, the top 2 of each group plus the **8 best third-placed teams**
advance to the Round of 32. The 12 third-placed teams are ranked by overall criteria
(no head-to-head — they didn't play each other) in the following order:

    1. Points
    2. Overall goal difference
    3. Overall goals scored
    4. Conduct score      — NOT IMPLEMENTED; needs yellow/red-card data
    5. FIFA ranking       — used if provided, else skipped
    6. Drawing of lots    — random, seeded by the rng argument

Top 8 advance, bottom 4 eliminated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from wc2026.sim.groups import GroupResult, TeamStanding

N_THIRDS = 12
N_ADVANCING = 8


@dataclass(frozen=True)
class ThirdPlacedEntry:
    """One team in the cross-group third-placed ranking."""

    team: str
    group: str
    standing: TeamStanding
    rank: int  # 1..12; 1..8 advance


def rank_third_placed(
    group_results: Mapping[str, GroupResult],
    rng: np.random.Generator,
    *,
    fifa_ranking: dict[str, int] | None = None,
) -> tuple[ThirdPlacedEntry, ...]:
    """Rank the 12 third-placed teams; return all 12 in finishing order, top first.

    Caller uses ``entries[:8]`` for those advancing to R32 and ``entries[8:]`` for
    those eliminated.
    """
    if len(group_results) != N_THIRDS:
        raise ValueError(f"expected {N_THIRDS} group results, got {len(group_results)}")
    fifa = fifa_ranking or {}

    thirds: list[tuple[str, TeamStanding]] = []
    for letter, gr in group_results.items():
        if len(gr.standings) != 4:
            raise ValueError(f"group {letter} has {len(gr.standings)} standings, expected 4")
        third = gr.standings[2]  # 0-indexed: 0=1st, 1=2nd, 2=3rd
        thirds.append((letter, third))

    # Sort by primary key (descending in pts/gd/gs).
    def primary_key(item: tuple[str, TeamStanding]) -> tuple[int, int, int]:
        _, s = item
        return (-s.points, -s.goal_difference, -s.goals_for)

    primary_sorted = sorted(thirds, key=primary_key)

    # Break ties: FIFA ranking → drawing of lots (no H2H since groups don't overlap).
    final: list[tuple[str, TeamStanding]] = []
    i = 0
    while i < len(primary_sorted):
        j = i + 1
        key_i = primary_key(primary_sorted[i])
        while j < len(primary_sorted) and primary_key(primary_sorted[j]) == key_i:
            j += 1
        tied = primary_sorted[i:j]
        if len(tied) == 1:
            final.append(tied[0])
        else:
            final.extend(
                sorted(
                    tied,
                    key=lambda item: (fifa.get(item[1].team, 10**9), rng.random()),
                )
            )
        i = j

    return tuple(
        ThirdPlacedEntry(team=s.team, group=letter, standing=s, rank=rank)
        for rank, (letter, s) in enumerate(final, start=1)
    )
