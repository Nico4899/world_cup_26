"""Parse the 72 NULL-scored Jürisoo rows into a WC 2026 fixture structure.

The Jürisoo dataset's WC 2026 rows give us:
    date, home_team, away_team, city, country, neutral, tournament="FIFA World Cup"

…but they don't carry FIFA's group letters. We recover groups by **clique detection**
on the match graph: each group of 4 forms a K_4 (every team plays every other team in
the group exactly once), so the 4 teams that share matches with each other are the
group.

Group letters A-L are then assigned by ordering on each group's earliest fixture date,
which matches the FIFA convention (Group A opens the tournament on day 1, Group B
typically opens on day 2, and so on).
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field

import pandas as pd

EXPECTED_GROUPS = 12
EXPECTED_TEAMS_PER_GROUP = 4
EXPECTED_MATCHES = 72
EXPECTED_MATCHES_PER_TEAM = 3
GROUP_LETTERS: tuple[str, ...] = tuple(string.ascii_uppercase[:EXPECTED_GROUPS])  # A..L


@dataclass(frozen=True)
class FixtureMatch:
    """One scheduled group-stage match."""

    date: pd.Timestamp
    home_team: str
    away_team: str
    group: str  # "A" .. "L"
    city: str
    country: str  # USA / Mexico / Canada
    neutral: bool


@dataclass(frozen=True)
class WC2026Fixtures:
    """Structured view of the 12-group / 72-match group stage."""

    groups: dict[str, tuple[str, ...]]  # {"A": (team1, team2, team3, team4), ...}
    matches: tuple[FixtureMatch, ...] = field()

    @property
    def teams(self) -> tuple[str, ...]:
        return tuple(team for group_teams in self.groups.values() for team in group_teams)

    def group_of(self, team: str) -> str:
        for letter, members in self.groups.items():
            if team in members:
                return letter
        raise KeyError(f"team {team!r} not in any group")

    def matches_in_group(self, group: str) -> tuple[FixtureMatch, ...]:
        return tuple(m for m in self.matches if m.group == group)


def parse_wc2026_fixtures(scheduled: pd.DataFrame) -> WC2026Fixtures:
    """Build a WC2026Fixtures from the Jürisoo NULL-scored rows.

    Expects ``scheduled`` to be the output of ``load_scheduled()`` filtered to
    WC 2026 (tournament == "FIFA World Cup"). Validates that exactly 72 matches
    yield exactly 12 four-team cliques and that every team plays exactly 3 group games.
    """
    df = scheduled[scheduled["tournament"] == "FIFA World Cup"].copy()
    if len(df) != EXPECTED_MATCHES:
        raise ValueError(f"expected {EXPECTED_MATCHES} WC 2026 fixtures, got {len(df)}")

    # Build adjacency: team -> set of opponents in the group stage.
    opponents: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        opponents.setdefault(h, set()).add(a)
        opponents.setdefault(a, set()).add(h)

    # Each team must play exactly 3 opponents (the other 3 members of its group).
    bad_degree = {t: len(opps) for t, opps in opponents.items() if len(opps) != EXPECTED_MATCHES_PER_TEAM}
    if bad_degree:
        raise ValueError(
            f"every team must play {EXPECTED_MATCHES_PER_TEAM} opponents; "
            f"violations: {bad_degree}"
        )

    # Each team plus its 3 opponents forms a group of 4 — but we need to verify
    # all 4 are mutually connected (i.e. a K_4 clique).
    groups_raw: list[tuple[str, ...]] = []
    assigned: set[str] = set()
    for team in sorted(opponents.keys()):
        if team in assigned:
            continue
        members = {team, *opponents[team]}
        if len(members) != EXPECTED_TEAMS_PER_GROUP:
            raise ValueError(f"group around {team!r} has {len(members)} members, expected 4")
        # K_4 check
        for m in members:
            if (members - {m}) != opponents[m]:
                raise ValueError(
                    f"group around {team!r} is not a K_4 clique; team {m!r} has "
                    f"opponents {opponents[m]} but group is {members}"
                )
        groups_raw.append(tuple(sorted(members)))
        assigned.update(members)

    if len(groups_raw) != EXPECTED_GROUPS:
        raise ValueError(f"derived {len(groups_raw)} groups, expected {EXPECTED_GROUPS}")

    # Label groups A-L by earliest fixture date (FIFA convention: Group A opens the
    # tournament). Ties are broken by the canonical sorted-tuple of team names.
    first_date = {}
    for group_members in groups_raw:
        members_set = set(group_members)
        earliest = df[
            df["home_team"].isin(members_set) & df["away_team"].isin(members_set)
        ]["date"].min()
        first_date[group_members] = earliest
    groups_raw_sorted = sorted(groups_raw, key=lambda g: (first_date[g], g))
    groups: dict[str, tuple[str, ...]] = {
        letter: members for letter, members in zip(GROUP_LETTERS, groups_raw_sorted, strict=True)
    }

    # Build FixtureMatch list with group letter populated.
    team_to_group: dict[str, str] = {
        team: letter for letter, members in groups.items() for team in members
    }
    matches: list[FixtureMatch] = []
    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        g_h, g_a = team_to_group[h], team_to_group[a]
        if g_h != g_a:
            # Already protected by the K_4 check, but belt-and-braces.
            raise ValueError(f"intra-group match {h!r} vs {a!r} spans groups {g_h} and {g_a}")
        matches.append(
            FixtureMatch(
                date=row["date"],
                home_team=h,
                away_team=a,
                group=g_h,
                city=str(row["city"]),
                country=str(row["country"]),
                neutral=bool(row["neutral"]),
            )
        )

    return WC2026Fixtures(groups=groups, matches=tuple(matches))
