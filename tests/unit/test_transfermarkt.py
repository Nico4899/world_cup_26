"""Tests for the polite Transfermarkt squad-market-value scraper."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from urllib.robotparser import RobotFileParser

import pandas as pd
import pytest
import requests

from wc2026.ingest import transfermarkt as tm


# --- parse_market_value ------------------------------------------------------


def test_parse_market_value_handles_billions() -> None:
    assert tm.parse_market_value("€1.05bn") == pytest.approx(1.05e9)
    assert tm.parse_market_value("€892.50m") == pytest.approx(8.925e8)
    assert tm.parse_market_value("€500k") == pytest.approx(5e5)


def test_parse_market_value_handles_plain_euros() -> None:
    assert tm.parse_market_value("€42") == pytest.approx(42.0)


def test_parse_market_value_returns_none_on_non_match() -> None:
    assert tm.parse_market_value("not a price") is None
    assert tm.parse_market_value("") is None


def test_parse_market_value_is_case_insensitive_for_suffix() -> None:
    assert tm.parse_market_value("€1.5M") == pytest.approx(1_500_000.0)
    assert tm.parse_market_value("€2.0BN") == pytest.approx(2_000_000_000.0)


# --- parse_squad_page --------------------------------------------------------


def test_parse_squad_page_reads_total_market_value() -> None:
    html = """
    <html>
      <head><title>Argentina | National team | Transfermarkt</title></head>
      <body>
        <div>Total market value: <a>€892.50m</a></div>
      </body>
    </html>
    """
    row = tm.parse_squad_page(html, team_slug="argentinien")
    assert row is not None
    assert row.team_slug == "argentinien"
    assert row.team_name == "Argentina"
    assert row.squad_market_value_eur == pytest.approx(8.925e8)


def test_parse_squad_page_falls_back_to_first_euro_value() -> None:
    """No 'Total market value' label, but a single € figure on the page."""
    html = "<html><head><title>Brazil | NT</title></head><body>€1.20bn squad</body></html>"
    row = tm.parse_squad_page(html, team_slug="brasilien")
    assert row is not None
    assert row.squad_market_value_eur == pytest.approx(1.2e9)


def test_parse_squad_page_returns_none_on_no_euro_value() -> None:
    html = "<html><head><title>No team</title></head><body>just text</body></html>"
    assert tm.parse_squad_page(html, team_slug="x") is None


# --- write_snapshot + load_latest_snapshot ---------------------------------


def test_write_and_load_snapshot_round_trip(tmp_path: Path) -> None:
    rows = [
        tm.TeamMarketValue(
            team_slug="argentinien",
            team_name="Argentina",
            squad_market_value_eur=8.925e8,
            snapshot_date=pd.Timestamp("2026-05-26"),
        ),
        tm.TeamMarketValue(
            team_slug="brasilien",
            team_name="Brazil",
            squad_market_value_eur=1.05e9,
            snapshot_date=pd.Timestamp("2026-05-26"),
        ),
    ]
    out = tm.write_snapshot(rows, tmp_path, today=datetime(2026, 5, 26, tzinfo=UTC))
    assert out.name == "squad_market_value_2026-05-26.parquet"
    df = tm.load_latest_snapshot(tmp_path)
    assert len(df) == 2
    assert set(df["team_name"]) == {"Argentina", "Brazil"}
    assert df["squad_market_value_eur"].sum() == pytest.approx(8.925e8 + 1.05e9)


def test_load_latest_snapshot_raises_when_dir_empty(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no transfermarkt snapshot"):
        tm.load_latest_snapshot(tmp_path)


# --- robots.txt fail-closed --------------------------------------------------


def test_fetch_team_market_value_skipped_when_robots_disallows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If robots.txt disallows our UA on the URL, return None (don't fetch)."""
    rp = RobotFileParser()
    rp.parse(["User-agent: *", "Disallow: /"])
    # No HTTP call should happen — fail loudly if it does.

    def boom(*_args: object, **_kw: object) -> object:
        raise AssertionError("HTTP call attempted despite robots.txt disallow")

    sess = requests.Session()
    monkeypatch.setattr(sess, "get", boom)
    got = tm.fetch_team_market_value(
        "https://www.transfermarkt.com/argentinien/startseite/verein/3437",
        session=sess,
        robots=rp,
    )
    assert got is None


def test_load_robots_fails_closed_on_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """When robots.txt itself is unreachable, the helper returns a parser that
    disallows everything — the polite default."""

    sess = requests.Session()

    def boom(*_args: object, **_kw: object) -> object:
        raise requests.ConnectionError("upstream offline")

    monkeypatch.setattr(sess, "get", boom)
    rp = tm._load_robots(sess)
    # Disallow-all rule means can_fetch returns False for anything.
    assert rp.can_fetch(tm.USER_AGENT, "https://www.transfermarkt.com/any") is False


def test_fetch_team_market_value_swallows_http_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 5xx (or a connection refused) should return None, not raise."""
    rp = RobotFileParser()
    rp.parse(["User-agent: *", "Allow: /"])  # permissive
    sess = requests.Session()

    def boom(*_args: object, **_kw: object) -> object:
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(sess, "get", boom)
    got = tm.fetch_team_market_value(
        "https://www.transfermarkt.com/x",
        session=sess,
        robots=rp,
    )
    assert got is None


# --- fetch_squad_market_values orchestrator ---------------------------------


def test_fetch_squad_market_values_writes_parquet_with_display_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The orchestrator replaces the URL-derived team_slug with the operator-
    supplied display name so downstream joins line up."""

    rp = RobotFileParser()
    rp.parse(["User-agent: *", "Allow: /"])
    monkeypatch.setattr(tm, "_load_robots", lambda _sess: rp)

    def fake_fetch(url: str, *_a: object, **_kw: object) -> tm.TeamMarketValue | None:
        return tm.TeamMarketValue(
            team_slug="from-url",
            team_name="From URL",
            squad_market_value_eur=1.0e8,
            snapshot_date=pd.Timestamp("2026-05-26"),
        )

    monkeypatch.setattr(tm, "fetch_team_market_value", fake_fetch)
    out = tm.fetch_squad_market_values(
        {"Argentina": "https://www.transfermarkt.com/argentinien/x"},
        target_dir=tmp_path,
        today=datetime(2026, 5, 26, tzinfo=UTC),
    )
    assert out is not None
    df = pd.read_parquet(out)
    assert df["team_name"].iloc[0] == "Argentina"


def test_fetch_squad_market_values_returns_none_when_all_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rp = RobotFileParser()
    rp.parse(["User-agent: *", "Allow: /"])
    monkeypatch.setattr(tm, "_load_robots", lambda _sess: rp)
    monkeypatch.setattr(tm, "fetch_team_market_value", lambda *_a, **_kw: None)
    out = tm.fetch_squad_market_values(
        {"Argentina": "https://x"},
        target_dir=tmp_path,
    )
    assert out is None
