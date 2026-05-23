"""Tests for the WC 2026 R32 bracket-resolution module."""

from __future__ import annotations

import pytest

from wc2026.sim.bracket import (
    FINAL_PAIR,
    QF_PAIRS,
    R16_PAIRS,
    R32_SLOTS,
    SF_PAIRS,
    assign_thirds_to_slots,
    resolve_r32_matchups,
    third_slot_eligibility,
)


def test_r32_has_16_slots() -> None:
    assert len(R32_SLOTS) == 16


def test_r32_match_ids_are_73_through_88() -> None:
    ids = sorted(mid for mid, _, _ in R32_SLOTS)
    assert ids == list(range(73, 89))


def test_r16_has_8_qf_4_sf_2() -> None:
    assert len(R16_PAIRS) == 8
    assert len(QF_PAIRS) == 4
    assert len(SF_PAIRS) == 2
    # final is one match (the third-place playoff also exists, but FINAL_PAIR is the final only)
    assert FINAL_PAIR[0] == 104


def test_each_round_consumes_each_winner_exactly_once() -> None:
    """The winners of R32 must each feed exactly one R16; R16 each one QF; etc."""
    r32_ids = {mid for mid, _, _ in R32_SLOTS}
    fed_into_r16 = []
    for _, a, b in R16_PAIRS:
        fed_into_r16.extend([a, b])
    assert sorted(fed_into_r16) == sorted(r32_ids)

    r16_ids = {mid for mid, _, _ in R16_PAIRS}
    fed_into_qf = []
    for _, a, b in QF_PAIRS:
        fed_into_qf.extend([a, b])
    assert sorted(fed_into_qf) == sorted(r16_ids)

    qf_ids = {mid for mid, _, _ in QF_PAIRS}
    fed_into_sf = []
    for _, a, b in SF_PAIRS:
        fed_into_sf.extend([a, b])
    assert sorted(fed_into_sf) == sorted(qf_ids)

    sf_ids = {mid for mid, _, _ in SF_PAIRS}
    fed_into_final = [FINAL_PAIR[1], FINAL_PAIR[2]]
    assert sorted(fed_into_final) == sorted(sf_ids)


def test_eight_slots_take_a_third() -> None:
    elig = third_slot_eligibility()
    assert len(elig) == 8
    # Every eligible set should be size 5 (per FIFA's published constraints).
    assert all(len(s) == 5 for s in elig.values())


def test_assign_thirds_succeeds_for_all_groups_advancing() -> None:
    """If thirds from A, B, C, D, E, F, G, H advance, greedy must produce a valid assignment."""
    advancing = {letter: f"team_{letter}" for letter in "ABCDEFGH"}
    elig = third_slot_eligibility()
    assignment = assign_thirds_to_slots(advancing, elig)
    assert len(assignment) == 8
    # Every assigned team is one of the advancing teams.
    assert set(assignment.values()) == set(advancing.values())
    # Each (slot, team) pair respects the eligibility constraint.
    for mid, team in assignment.items():
        # team_X corresponds to advancing key X
        group = next(g for g, t in advancing.items() if t == team)
        assert group in elig[mid], f"slot {mid} got team from group {group} (not in {sorted(elig[mid])})"


def test_assign_thirds_succeeds_for_alternative_advancing_sets() -> None:
    """Spot-check a few non-trivial sets of advancing groups."""
    elig = third_slot_eligibility()
    for letters in [
        "ABCDFGIJ",  # mix
        "CDEFGHIJ",  # latter half
        "ABCEHIJK",  # mostly early letters but skipping
        "BDFHJL",  # only 6 — should fail validation
    ]:
        if len(letters) != 8:
            with pytest.raises(ValueError, match="expected 8 advancing thirds"):
                assign_thirds_to_slots({lt: f"t_{lt}" for lt in letters}, elig)
            continue
        advancing = {lt: f"t_{lt}" for lt in letters}
        assignment = assign_thirds_to_slots(advancing, elig)
        assert len(assignment) == 8
        assert set(assignment.values()) == set(advancing.values())


def test_resolve_r32_full_matchups() -> None:
    winners = {letter: f"W{letter}" for letter in "ABCDEFGHIJKL"}
    runners = {letter: f"R{letter}" for letter in "ABCDEFGHIJKL"}
    advancing = {letter: f"T{letter}" for letter in "ABCDEFGH"}
    matchups = resolve_r32_matchups(winners, runners, advancing)
    assert len(matchups) == 16
    # Every team across all matchups should be one of the qualifiers (32 distinct).
    teams = {t for pair in matchups.values() for t in pair}
    assert len(teams) == 32
    expected = set(winners.values()) | set(runners.values()) | set(advancing.values())
    assert teams == expected
    # Spot-check: Match 73 must be 2A vs 2B per the slot definition.
    assert matchups[73] == (runners["A"], runners["B"])
    # Spot-check: Match 75 = 1F vs 2C
    assert matchups[75] == (winners["F"], runners["C"])


def test_every_12_choose_8_advancing_subset_has_a_valid_matching() -> None:
    """Exhaustive: for ALL 495 possible 8-of-12 advancing-thirds sets, the bipartite
    matcher must find an assignment. If this fails, the encoded eligibility sets
    disagree with FIFA's published constraint table for some scenario."""
    from itertools import combinations

    elig = third_slot_eligibility()
    groups = "ABCDEFGHIJKL"
    failed: list[tuple[str, ...]] = []
    for subset in combinations(groups, 8):
        advancing = {g: f"t_{g}" for g in subset}
        try:
            assignment = assign_thirds_to_slots(advancing, elig)
        except ValueError:
            failed.append(subset)
            continue
        # validate: every assignment respects eligibility
        for mid, team in assignment.items():
            group = next(g for g, t in advancing.items() if t == team)
            assert group in elig[mid]
    assert not failed, f"{len(failed)}/495 subsets unsolvable; first few: {failed[:5]}"


def test_resolve_r32_rejects_incomplete_inputs() -> None:
    winners = {letter: f"W{letter}" for letter in "ABCDEFGHIJK"}  # missing L
    runners = {letter: f"R{letter}" for letter in "ABCDEFGHIJKL"}
    advancing = {letter: f"T{letter}" for letter in "ABCDEFGH"}
    with pytest.raises(ValueError, match="group_winners"):
        resolve_r32_matchups(winners, runners, advancing)
