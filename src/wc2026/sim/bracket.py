"""R32 → final bracket structure for the 2026 World Cup.

Source: en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage (fetched 2026-05-23).
The Wikipedia bracket gives 16 R32 slots, eight of which take a group winner vs a
third-placed team. For each of those eight slots, FIFA's regulations limit the
eligible source groups for the third-placed team (so the bracket avoids putting
a group winner against a third-placed team from the same group, and tries to keep
top-seeded teams apart on opposite sides of the bracket).

This module encodes:

  * ``R32_SLOTS`` — the 16 (left_descriptor, right_descriptor) slot definitions
  * ``assign_thirds_to_slots`` — bipartite assignment of the 8 advancing thirds
    to the 8 "3rd-from-{set}" slots, respecting each slot's eligible-group set
  * ``R16_PAIRS`` / ``QF_PAIRS`` / ``SF_PAIRS`` / ``FINAL_PAIR`` — how the slot
    indices propagate through the bracket

Simplification (documented limitation)
--------------------------------------
FIFA publishes a precomputed 12-choose-8 lookup table that maps each possible set
of 8 advancing groups to a specific third-to-slot permutation. We instead solve
the bipartite assignment greedily by most-constrained-first. This produces a valid
assignment in 100% of tested scenarios; whether it matches FIFA's exact published
permutation in every case is **not verified** and should be checked against the
official 2026 regulations document before any public claim is made.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SlotKind = Literal["winner", "runner_up", "third"]


@dataclass(frozen=True)
class SlotDescriptor:
    """Specifies which team fills one half of an R32 matchup."""

    kind: SlotKind
    group: str | None = None  # "A".."L" for winner/runner_up; None for third
    eligible_groups: frozenset[str] | None = None  # set of "A".."L" for third slots


def _winner(group: str) -> SlotDescriptor:
    return SlotDescriptor(kind="winner", group=group)


def _runner_up(group: str) -> SlotDescriptor:
    return SlotDescriptor(kind="runner_up", group=group)


def _third(eligible: str) -> SlotDescriptor:
    return SlotDescriptor(kind="third", eligible_groups=frozenset(eligible))


# Match numbers 73-88 are the R32 matches per FIFA numbering.
# Each entry is (match_id, left_slot, right_slot).
R32_SLOTS: tuple[tuple[int, SlotDescriptor, SlotDescriptor], ...] = (
    (73, _runner_up("A"), _runner_up("B")),
    (74, _winner("E"), _third("ABCDF")),
    (75, _winner("F"), _runner_up("C")),
    (76, _winner("C"), _runner_up("F")),
    (77, _winner("I"), _third("CDFGH")),
    (78, _runner_up("E"), _runner_up("I")),
    (79, _winner("A"), _third("CEFHI")),
    (80, _winner("L"), _third("EHIJK")),
    (81, _winner("D"), _third("BEFIJ")),
    (82, _winner("G"), _third("AEHIJ")),
    (83, _runner_up("K"), _runner_up("L")),
    (84, _winner("H"), _runner_up("J")),
    (85, _winner("B"), _third("EFGIJ")),
    (86, _winner("J"), _runner_up("H")),
    (87, _winner("K"), _third("DEIJL")),
    (88, _runner_up("D"), _runner_up("G")),
)

# Round of 16: winners of (match_a, match_b) → r16 match. From Wikipedia bracket.
R16_PAIRS: tuple[tuple[int, int, int], ...] = (
    (89, 73, 75),
    (90, 79, 81),
    (91, 74, 76),
    (92, 80, 82),
    (93, 77, 78),
    (94, 87, 88),
    (95, 83, 84),
    (96, 85, 86),
)

QF_PAIRS: tuple[tuple[int, int, int], ...] = (
    (97, 89, 90),
    (98, 91, 92),
    (99, 93, 94),
    (100, 95, 96),
)

SF_PAIRS: tuple[tuple[int, int, int], ...] = (
    (101, 97, 98),
    (102, 99, 100),
)

FINAL_PAIR: tuple[int, int, int] = (104, 101, 102)
# Third-place playoff is match 103, between the two SF losers.
THIRD_PLACE_PAIR: tuple[int, int, int] = (103, 101, 102)


def _find_matching(
    slot_ids: list[int],
    eligibility: dict[int, frozenset[str]],
    remaining: frozenset[str],
) -> dict[int, str] | None:
    """Recursive backtracking bipartite matching, most-constrained-slot-first."""
    if not slot_ids:
        return {}
    # Pick the slot with fewest remaining options to fail fast.
    ranked = sorted(
        slot_ids,
        key=lambda sid: (len(eligibility[sid] & remaining), sid),
    )
    sid = ranked[0]
    others = [s for s in slot_ids if s != sid]
    candidates = sorted(eligibility[sid] & remaining)
    for g in candidates:
        sub = _find_matching(others, eligibility, remaining - {g})
        if sub is not None:
            return {sid: g, **sub}
    return None


def assign_thirds_to_slots(
    advancing: dict[str, str],
    slot_eligibility: dict[int, frozenset[str]],
) -> dict[int, str]:
    """Assign each of the 8 'third' R32 slots a team from the 8 advancing thirds.

    Uses backtracking bipartite matching (most-constrained-slot-first), which is
    exact and finds an assignment whenever one exists. For the published 2026
    eligibility sets this returns in microseconds; if no matching exists we raise
    ValueError (indicates either a logic bug or that the encoded sets disagree
    with FIFA's regulations document).
    """
    if len(advancing) != 8:
        raise ValueError(f"expected 8 advancing thirds, got {len(advancing)}")
    if len(slot_eligibility) != 8:
        raise ValueError(f"expected 8 third-slot eligibility sets, got {len(slot_eligibility)}")

    assignment_groups = _find_matching(
        list(slot_eligibility.keys()),
        slot_eligibility,
        frozenset(advancing.keys()),
    )
    if assignment_groups is None:
        raise ValueError(
            "no valid bipartite matching: advancing thirds "
            f"{sorted(advancing.keys())} cannot fill the 8 third-slot eligibility sets"
        )
    return {sid: advancing[g] for sid, g in assignment_groups.items()}


def third_slot_eligibility() -> dict[int, frozenset[str]]:
    """Return ``{match_id: eligible_groups}`` for the 8 R32 slots that take a third."""
    return {
        mid: right.eligible_groups
        for mid, _, right in R32_SLOTS
        if right.kind == "third" and right.eligible_groups is not None
    }


def resolve_r32_matchups(
    group_winners: dict[str, str],
    runners_up: dict[str, str],
    advancing_thirds: dict[str, str],
) -> dict[int, tuple[str, str]]:
    """Resolve the 16 R32 matchups into concrete (team_a, team_b) pairs.

    Parameters
    ----------
    group_winners :
        ``{group_letter: team_name}`` for all 12 group winners.
    runners_up :
        ``{group_letter: team_name}`` for all 12 group runners-up.
    advancing_thirds :
        ``{group_letter: team_name}`` for the 8 advancing third-placed teams.
    """
    if set(group_winners.keys()) != set("ABCDEFGHIJKL"):
        raise ValueError("group_winners must cover all groups A-L")
    if set(runners_up.keys()) != set("ABCDEFGHIJKL"):
        raise ValueError("runners_up must cover all groups A-L")
    if len(advancing_thirds) != 8:
        raise ValueError(f"expected 8 advancing thirds, got {len(advancing_thirds)}")

    third_for_slot = assign_thirds_to_slots(advancing_thirds, third_slot_eligibility())

    out: dict[int, tuple[str, str]] = {}
    for mid, left, right in R32_SLOTS:
        a = _resolve(left, group_winners, runners_up, third_for_slot, mid)
        b = _resolve(right, group_winners, runners_up, third_for_slot, mid)
        out[mid] = (a, b)
    return out


def _resolve(
    slot: SlotDescriptor,
    winners: dict[str, str],
    runners_up: dict[str, str],
    third_for_slot: dict[int, str],
    match_id: int,
) -> str:
    if slot.kind == "winner":
        assert slot.group is not None
        return winners[slot.group]
    if slot.kind == "runner_up":
        assert slot.group is not None
        return runners_up[slot.group]
    return third_for_slot[match_id]
