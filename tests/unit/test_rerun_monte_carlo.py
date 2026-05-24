"""Unit tests for scripts/rerun_monte_carlo.py."""

from __future__ import annotations

from datetime import date

import pytest
import scripts.rerun_monte_carlo as mcr
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from wc2026.db.models import Base, RawLiveEvent, TournamentSimRun, TournamentSimTeamOutcome


@pytest.fixture
def sqlite_engine():
    eng = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(eng)
    try:
        yield eng
    finally:
        eng.dispose()


def _seed_ft_whistle(engine, *, match_id: int, h: int, a: int) -> None:
    from datetime import UTC, datetime

    with Session(engine, future=True) as session:
        session.add(
            RawLiveEvent(
                match_id=match_id,
                seq=99,
                minute=90,
                period=2,
                event_type="FT_WHISTLE",
                team=None,
                player=None,
                home_score_after=h,
                away_score_after=a,
                home_red_cards_after=0,
                away_red_cards_after=0,
                ingested_at=datetime.now(UTC),
            )
        )
        session.commit()


def test_rerun_and_persist_skips_without_database_url(monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("WC2026_DATABASE_URL", raising=False)
    assert mcr.rerun_and_persist() is None


def test_rerun_and_persist_skips_without_artefact(monkeypatch, sqlite_engine, tmp_path) -> None:
    """Engine provided so the env check is skipped — but no artefact on disk."""
    run_id = mcr.rerun_and_persist(
        artefact_path=tmp_path / "missing.npz",
        engine=sqlite_engine,
    )
    assert run_id is None


def test_persist_run_writes_run_plus_team_outcomes(sqlite_engine) -> None:
    """Build a tiny TournamentSummary by hand and verify the persistence shape."""
    import pandas as pd

    from wc2026.sim.tournament import ROUND_COLUMNS, TournamentSummary

    teams = ["Mexico", "Senegal", "Argentina"]
    # ROUND_COLUMNS layout (Phase 12+): group_winner, runner_up, third_advance,
    # third_out, fourth, r32_reached, r16_reached, qf_reached, sf_reached,
    # final_reached, champion.
    df = pd.DataFrame(
        [[0.5, 0.3, 0.2, 0.1, 0.1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]] * len(teams),
        index=teams,
        columns=list(ROUND_COLUMNS),
    )
    df.index.name = "team"
    summary = TournamentSummary(n_sims=100, probabilities=df)
    run_id = mcr.persist_run(engine=sqlite_engine, summary=summary, model_version="test.v1")
    assert isinstance(run_id, int)
    with sqlite_engine.connect() as conn:
        runs = list(conn.execute(select(TournamentSimRun)).all())
        outcomes = list(conn.execute(select(TournamentSimTeamOutcome)).all())
    assert len(runs) == 1
    assert len(outcomes) == len(teams)


def test_rerun_and_persist_writes_to_db_with_no_known_results(
    monkeypatch, sqlite_engine, tmp_path
) -> None:
    """A full end-to-end with monkey-patched fixtures + a hand-written tiny artefact.

    We mock the heavy fit by patching ``_hydrate_model`` to return a mock model
    whose ``score_probs`` is deterministic; the simulator and persistence ride
    on top.
    """
    import numpy as np

    from wc2026.sim import tournament as tournament_mod

    # A fake PoissonDC whose .score_probs always returns mass on (1,0).
    class _FakeModel:
        def score_probs(self, home, away, *, neutral=False):
            _ = home, away, neutral
            p = np.zeros((11, 11))
            p[1, 0] = 1.0
            return p

        def outcome_probs(self, home, away, *, neutral=False):
            _ = home, away, neutral
            return {"home_win": 1.0, "draw": 0.0, "away_win": 0.0}

        def expected_goals(self, home, away, *, neutral=False):
            _ = home, away, neutral
            return (1.0, 0.0)

    monkeypatch.setattr(mcr, "hydrate_from_artefact", lambda *_a, **_k: _FakeModel())
    monkeypatch.setattr(mcr, "load_wc_match_id_map", lambda: {})

    # Pretend the artefact exists so the pre-flight check passes.
    artefact = tmp_path / "model.npz"
    artefact.write_bytes(b"\x00")

    # The simulator uses real fixtures; patch fixtures load too.
    fixtures = mcr.load_wc2026_fixtures()

    monkeypatch.setattr(mcr, "load_wc2026_fixtures", lambda: fixtures)

    # Use a small n_sims so the test stays fast.
    run_id = mcr.rerun_and_persist(
        n_sims=10,
        seed=0,
        artefact_path=artefact,
        engine=sqlite_engine,
    )
    assert isinstance(run_id, int)
    with sqlite_engine.connect() as conn:
        n_outcomes = len(conn.execute(select(TournamentSimTeamOutcome)).all())
    # 48 teams in the WC 2026 fixtures.
    assert n_outcomes == 48
    _ = tournament_mod  # keep import alive for type-narrowing


def test_rerun_loads_known_results_from_live_events(monkeypatch, sqlite_engine, tmp_path) -> None:
    """If a FT_WHISTLE row exists and the mapping covers it, the run picks it up."""
    import numpy as np

    class _FakeModel:
        def score_probs(self, home, away, *, neutral=False):
            _ = home, away, neutral
            p = np.zeros((11, 11))
            p[1, 0] = 1.0
            return p

        def outcome_probs(self, home, away, *, neutral=False):
            return {"home_win": 1.0, "draw": 0.0, "away_win": 0.0}

        def expected_goals(self, home, away, *, neutral=False):
            return (1.0, 0.0)

    monkeypatch.setattr(mcr, "hydrate_from_artefact", lambda *_a, **_k: _FakeModel())
    fixtures = mcr.load_wc2026_fixtures()
    monkeypatch.setattr(mcr, "load_wc2026_fixtures", lambda: fixtures)
    # Map FDO match_id 1 → the first WC 2026 fixture (Mexico vs Senegal).
    first = fixtures.matches[0]
    mapping = {1: (date(2026, 6, 11), first.home_team, first.away_team)}
    monkeypatch.setattr(mcr, "load_wc_match_id_map", lambda: mapping)
    # Seed a FT_WHISTLE row so the known-results map is non-empty.
    _seed_ft_whistle(sqlite_engine, match_id=1, h=4, a=0)

    artefact = tmp_path / "model.npz"
    artefact.write_bytes(b"\x00")

    captured_known: dict = {}

    real_mc = mcr.simulate_tournament_monte_carlo

    def spy_mc(*args, **kwargs):
        captured_known.update(kwargs.get("known_group_results", {}))
        return real_mc(*args, **kwargs)

    monkeypatch.setattr(mcr, "simulate_tournament_monte_carlo", spy_mc)

    mcr.rerun_and_persist(n_sims=5, seed=0, artefact_path=artefact, engine=sqlite_engine)

    # The conditioning bridged through correctly: known_group_results contains
    # the (home, away) → (4, 0) lock-in.
    assert captured_known == {(first.home_team, first.away_team): (4, 0)}
