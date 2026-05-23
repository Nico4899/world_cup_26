"""Host-team flags for WC 2026 (USA / Mexico / Canada).

The 2026 World Cup is co-hosted by three nations. Whichever side of the
matchup involves a host team gets a small home-advantage bonus in the
literature (Pollard 2008; Goller & Heiniger 2022). We expose this as two
0/1 flags rather than a single signed indicator so the model can learn
asymmetric effects (e.g. host nations may concede more under pressure).
"""

from __future__ import annotations

# The three WC 2026 host nations. Stored as a frozenset because membership
# checks are the only operation we ever do on this collection.
WC_2026_HOSTS: frozenset[str] = frozenset({"United States", "Mexico", "Canada"})


def is_host(team: str) -> int:
    """1 if ``team`` is a WC 2026 host nation, else 0."""
    return int(team in WC_2026_HOSTS)


__all__ = ["WC_2026_HOSTS", "is_host"]
