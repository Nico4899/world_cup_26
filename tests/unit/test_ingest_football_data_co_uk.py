"""Unit tests for the football-data.co.uk CSV ingester."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from wc2026.ingest.football_data_co_uk import (
    fetch_calibration_corpus,
    fetch_league_csv,
    implied_probabilities,
    parse_csv,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


def _fixture_text() -> str:
    return (FIXTURE_DIR / "football_data_co_uk_sample.csv").read_text(encoding="utf-8")


def test_parse_csv_extracts_four_matches() -> None:
    df = parse_csv(_fixture_text())
    assert len(df) == 4
    assert list(df["home_team"]) == ["Manchester Utd", "Ipswich", "Arsenal", "Brighton"]


def test_parse_csv_typed_scores_and_result() -> None:
    df = parse_csv(_fixture_text())
    assert df["fthg"].tolist() == [1, 0, 2, 3]
    assert df["ftag"].tolist() == [0, 2, 0, 0]
    assert list(df["ftr"]) == ["H", "A", "H", "H"]


def test_parse_csv_picks_pinnacle_when_b365_missing() -> None:
    df = parse_csv(_fixture_text())
    # Arsenal vs Wolves: B365C* are blank in the fixture → PC* (Pinnacle) used.
    arsenal_row = df[df["home_team"] == "Arsenal"].iloc[0]
    assert arsenal_row["odds_source"] == "PC"
    assert arsenal_row["odds_home"] == 1.30


def test_parse_csv_marks_rows_with_no_odds() -> None:
    df = parse_csv(_fixture_text())
    # Brighton row: B365C* set, PC* blank → uses Bet365.
    brighton_row = df[df["home_team"] == "Brighton"].iloc[0]
    assert brighton_row["odds_source"] == "B365C"


def test_parse_csv_handles_empty_payload() -> None:
    df = parse_csv("")
    assert df.empty


def test_implied_probabilities_sum_to_one_per_row() -> None:
    df = implied_probabilities(parse_csv(_fixture_text()))
    sums = (df["p_home"] + df["p_draw"] + df["p_away"]).dropna()
    assert ((sums - 1.0).abs() < 1e-9).all()


def test_implied_probabilities_homes_track_short_prices() -> None:
    """A 1.50 home favourite must have higher implied home prob than a 9.00 dog."""
    df = implied_probabilities(parse_csv(_fixture_text()))
    mu_home = df[df["home_team"] == "Manchester Utd"]["p_home"].iloc[0]
    ipswich_home = df[df["home_team"] == "Ipswich"]["p_home"].iloc[0]
    assert mu_home > ipswich_home


class _StubResponse:
    def __init__(self, text: str):
        self.text = text
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None


class _StubSession:
    def __init__(self, responses: dict[str, str]):
        self._responses = responses
        self.headers: dict[str, str] = {}
        self.calls: list[str] = []

    def get(self, url: str, **_):
        self.calls.append(url)
        if url not in self._responses:
            raise RuntimeError(f"unexpected URL {url}")
        return _StubResponse(self._responses[url])


def test_fetch_league_csv_writes_parquet_under_season_league(tmp_path: Path) -> None:
    session = _StubSession({"https://www.football-data.co.uk/mmz4281/2425/E0.csv": _fixture_text()})
    out = fetch_league_csv(
        "2425",
        "E0",
        session=session,
        target_dir=tmp_path,
        today=datetime(2026, 5, 23, tzinfo=UTC),
    )
    assert out.name == "2425_E0_2026-05-23.parquet"
    df = pd.read_parquet(out)
    assert len(df) == 4


def test_fetch_calibration_corpus_returns_one_path_per_pair(tmp_path: Path) -> None:
    text = _fixture_text()
    session = _StubSession(
        {
            "https://www.football-data.co.uk/mmz4281/2425/E0.csv": text,
            "https://www.football-data.co.uk/mmz4281/2324/E0.csv": text,
        }
    )
    paths = fetch_calibration_corpus(
        [("2425", "E0"), ("2324", "E0")],
        session=session,
        target_dir=tmp_path,
        today=datetime(2026, 5, 23, tzinfo=UTC),
    )
    assert len(paths) == 2
    assert {p.name for p in paths} == {
        "2425_E0_2026-05-23.parquet",
        "2324_E0_2026-05-23.parquet",
    }
