"""Unit tests for the host-team flag helper."""

from __future__ import annotations

from wc2026.features.host_team import WC_2026_HOSTS, is_host


def test_wc_2026_hosts_set_contains_exactly_three_countries() -> None:
    assert WC_2026_HOSTS == frozenset({"United States", "Mexico", "Canada"})


def test_is_host_is_one_for_each_host_nation() -> None:
    assert is_host("United States") == 1
    assert is_host("Mexico") == 1
    assert is_host("Canada") == 1


def test_is_host_is_zero_for_non_hosts() -> None:
    for team in ("Argentina", "France", "Brazil", "Germany"):
        assert is_host(team) == 0


def test_is_host_distinguishes_us_from_usa_string() -> None:
    """Canonical name is 'United States'; the bare 'USA' string must not match."""
    assert is_host("USA") == 0
