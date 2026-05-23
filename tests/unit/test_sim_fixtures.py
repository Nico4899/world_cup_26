"""Tests for WC 2026 fixture parsing from Jürisoo NULL-scored rows."""

from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

from wc2026.ingest.kaggle_intl import load_scheduled
from wc2026.sim.fixtures import (
    EXPECTED_GROUPS,
    EXPECTED_MATCHES,
    EXPECTED_MATCHES_PER_TEAM,
    EXPECTED_TEAMS_PER_GROUP,
    GROUP_LETTERS,
    parse_wc2026_fixtures,
)


def _make_synthetic_fixtures() -> pd.DataFrame:
    """Build a 2-group, 12-match synthetic schedule (for shape-violation tests)."""
    matches = [
        # Group 1: AAA, BBB, CCC, DDD
        ("AAA", "BBB", "2026-06-11"),
        ("CCC", "DDD", "2026-06-11"),
        ("AAA", "CCC", "2026-06-15"),
        ("BBB", "DDD", "2026-06-15"),
        ("AAA", "DDD", "2026-06-19"),
        ("BBB", "CCC", "2026-06-19"),
        # Group 2: EEE, FFF, GGG, HHH
        ("EEE", "FFF", "2026-06-12"),
        ("GGG", "HHH", "2026-06-12"),
        ("EEE", "GGG", "2026-06-16"),
        ("FFF", "HHH", "2026-06-16"),
        ("EEE", "HHH", "2026-06-20"),
        ("FFF", "GGG", "2026-06-20"),
    ]
    rows = [
        {
            "date": pd.Timestamp(date),
            "home_team": h,
            "away_team": a,
            "home_score": pd.NA,
            "away_score": pd.NA,
            "tournament": "FIFA World Cup",
            "city": "Somewhere",
            "country": "United States",
            "neutral": True,
        }
        for h, a, date in matches
    ]
    return pd.DataFrame(rows)


# --- real-data tests --------------------------------------------------------


def test_parse_real_jurisoo_fixtures_has_12_groups_72_matches() -> None:
    fixtures = parse_wc2026_fixtures(load_scheduled())
    assert len(fixtures.groups) == EXPECTED_GROUPS
    assert len(fixtures.matches) == EXPECTED_MATCHES
    assert tuple(fixtures.groups.keys()) == GROUP_LETTERS


def test_every_team_plays_exactly_three_group_games() -> None:
    fixtures = parse_wc2026_fixtures(load_scheduled())
    counts: Counter[str] = Counter()
    for m in fixtures.matches:
        counts[m.home_team] += 1
        counts[m.away_team] += 1
    # 48 teams, each plays 3
    assert len(counts) == EXPECTED_GROUPS * EXPECTED_TEAMS_PER_GROUP
    assert all(n == EXPECTED_MATCHES_PER_TEAM for n in counts.values())


def test_every_group_has_four_teams_and_six_internal_matches() -> None:
    fixtures = parse_wc2026_fixtures(load_scheduled())
    for letter in GROUP_LETTERS:
        teams = fixtures.groups[letter]
        assert len(teams) == EXPECTED_TEAMS_PER_GROUP
        matches = fixtures.matches_in_group(letter)
        # 4 teams choose 2 = 6 distinct pairings → 6 matches
        n_expected = EXPECTED_TEAMS_PER_GROUP * (EXPECTED_TEAMS_PER_GROUP - 1) // 2
        assert len(matches) == n_expected
        # Every pair appears exactly once.
        pair_counts: Counter[tuple[str, str]] = Counter()
        for m in matches:
            pair = tuple(sorted([m.home_team, m.away_team]))
            pair_counts[pair] += 1
        assert all(c == 1 for c in pair_counts.values())


def test_group_a_contains_mexico_and_opens_the_tournament() -> None:
    fixtures = parse_wc2026_fixtures(load_scheduled())
    # FIFA convention: host (Mexico) is in Group A; opener is June 11.
    assert "Mexico" in fixtures.groups["A"]
    first_date = min(m.date for m in fixtures.matches_in_group("A"))
    assert first_date == pd.Timestamp("2026-06-11")
    # No other group should open earlier than Group A.
    for letter in GROUP_LETTERS[1:]:
        other_first = min(m.date for m in fixtures.matches_in_group(letter))
        assert other_first >= first_date


def test_host_country_matches_are_marked_not_neutral() -> None:
    """Mexico's, Canada's, and USA's group-stage matches in their home country must
    have neutral=False; all other matches must be neutral=True."""
    fixtures = parse_wc2026_fixtures(load_scheduled())
    host_team_to_country = {
        "Mexico": "Mexico",
        "Canada": "Canada",
        "United States": "United States",
    }
    for m in fixtures.matches:
        if m.home_team in host_team_to_country and m.country == host_team_to_country[m.home_team]:
            assert not m.neutral, f"{m.home_team} at home ({m.country}) should be non-neutral"
        else:
            assert m.neutral, (
                f"match {m.home_team} v {m.away_team} in {m.country} should be neutral"
            )


def test_group_of_lookup() -> None:
    fixtures = parse_wc2026_fixtures(load_scheduled())
    assert fixtures.group_of("Mexico") == "A"
    with pytest.raises(KeyError, match="not in any group"):
        fixtures.group_of("Nowhere")


def test_teams_property_lists_all_48() -> None:
    fixtures = parse_wc2026_fixtures(load_scheduled())
    teams = fixtures.teams
    assert len(teams) == EXPECTED_GROUPS * EXPECTED_TEAMS_PER_GROUP
    assert len(set(teams)) == len(teams)  # all distinct


# --- synthetic-schedule violation tests -------------------------------------


def test_parse_rejects_wrong_match_count() -> None:
    df = _make_synthetic_fixtures()  # 12 rows, but we expect 72
    with pytest.raises(ValueError, match=r"expected 72 WC 2026 fixtures"):
        parse_wc2026_fixtures(df)


def test_parse_rejects_team_with_wrong_degree() -> None:
    """If a team appears in <3 or >3 matches, fail loudly."""
    # Build 72 matches but make one team play 2 instead of 3 (and another play 4).
    df = load_scheduled().copy()
    # Swap one of Mexico's away opponents with another team's, breaking degrees.
    mexico_row = df[df["home_team"] == "Mexico"].index[0]
    df.loc[mexico_row, "home_team"] = "Brazil"  # now Mexico plays 2, Brazil plays 4
    with pytest.raises(ValueError, match="must play 3 opponents"):
        parse_wc2026_fixtures(df)
