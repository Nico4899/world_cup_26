"""Schema/relationship checks for wc2026.db.models.

Uses an in-memory SQLite engine so no Postgres is needed. SQLite ignores some
PG-specific features (e.g. ON DELETE CASCADE isn't enforced unless PRAGMA is set),
which is fine for verifying the model graph itself; integration tests cover the
actual Postgres behaviour.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session

from wc2026.db.models import (
    Base,
    MatchFeatures,
    ModelPrediction,
    RawEloSnapshot,
    RawFifaRanking,
    RawMatch,
    RawMatchXg,
    RawSquad,
    RawTeamAsset,
    SchedulerJobRun,
    TournamentSimRun,
    TournamentSimTeamOutcome,
)


@pytest.fixture
def sqlite_engine():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


def test_all_expected_tables_are_registered(sqlite_engine):
    insp = inspect(sqlite_engine)
    tables = set(insp.get_table_names())
    expected = {
        "raw_matches",
        "raw_elo_snapshots",
        "model_predictions",
        "tournament_sim_runs",
        "tournament_sim_team_outcomes",
        "scheduler_job_runs",
        "raw_team_assets",
        "raw_squads",
        "raw_fifa_rankings",
        "raw_match_xg",
        "features_match_features",
    }
    assert expected.issubset(tables), f"missing tables: {expected - tables}"


def test_raw_matches_columns_and_unique_constraint(sqlite_engine):
    insp = inspect(sqlite_engine)
    cols = {c["name"] for c in insp.get_columns("raw_matches")}
    assert {
        "id",
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
        "source",
        "ingested_at",
    }.issubset(cols)

    unique_names = {u["name"] for u in insp.get_unique_constraints("raw_matches")}
    assert "uq_raw_matches_natural_key" in unique_names


def test_raw_elo_snapshots_has_composite_pk(sqlite_engine):
    insp = inspect(sqlite_engine)
    pk = insp.get_pk_constraint("raw_elo_snapshots")
    assert set(pk["constrained_columns"]) == {"snapshot_date", "team_code"}


def test_tournament_sim_outcomes_has_fk_to_runs(sqlite_engine):
    insp = inspect(sqlite_engine)
    fks = insp.get_foreign_keys("tournament_sim_team_outcomes")
    assert len(fks) == 1
    fk = fks[0]
    assert fk["referred_table"] == "tournament_sim_runs"
    assert fk["referred_columns"] == ["run_id"]


def test_relationship_cascade_in_orm_layer(sqlite_engine):
    """Inserting outcomes via the parent .team_outcomes list should round-trip."""
    with Session(sqlite_engine) as s:
        run = TournamentSimRun(
            created_at=datetime.now(UTC), n_sims=1000, model_version="poisson-dc-v0"
        )
        run.team_outcomes.append(
            TournamentSimTeamOutcome(
                team="Argentina",
                group_winner_p=0.6,
                group_runner_up_p=0.3,
                advance_r32_p=0.9,
                advance_r16_p=0.7,
                quarterfinal_p=0.5,
                semifinal_p=0.3,
                final_p=0.2,
                champion_p=0.12,
            )
        )
        s.add(run)
        s.commit()

        loaded = s.scalar(select(TournamentSimRun))
        assert loaded is not None
        assert len(loaded.team_outcomes) == 1
        assert loaded.team_outcomes[0].team == "Argentina"
        assert loaded.team_outcomes[0].run is loaded


def test_can_insert_minimal_raw_match(sqlite_engine):
    with Session(sqlite_engine) as s:
        m = RawMatch(
            date=date(2026, 6, 11),
            home_team="Mexico",
            away_team="Senegal",
            tournament="FIFA World Cup",
            city="Mexico City",
            country="Mexico",
            neutral=False,
            source="jurisoo_kaggle",
            ingested_at=datetime.now(UTC),
        )
        s.add(m)
        s.commit()
        assert m.id is not None


def test_can_insert_model_prediction_with_json_payload(sqlite_engine):
    with Session(sqlite_engine) as s:
        p = ModelPrediction(
            match_date=date(2026, 6, 11),
            home_team="Mexico",
            away_team="Senegal",
            p_home=0.5,
            p_draw=0.25,
            p_away=0.25,
            score_matrix_json={"0-0": 0.08, "1-1": 0.11},
            model_version="poisson-dc-v0",
            created_at=datetime.now(UTC),
        )
        s.add(p)
        s.commit()
        loaded = s.scalar(select(ModelPrediction))
        assert loaded.score_matrix_json == {"0-0": 0.08, "1-1": 0.11}


def test_can_insert_scheduler_job_run(sqlite_engine):
    with Session(sqlite_engine) as s:
        r = SchedulerJobRun(
            job_name="kaggle_refresh",
            started_at=datetime.now(UTC),
            status="ok",
        )
        s.add(r)
        s.commit()
        assert r.id is not None


def test_elo_snapshot_round_trip(sqlite_engine):
    with Session(sqlite_engine) as s:
        snap = RawEloSnapshot(
            snapshot_date=date(2026, 5, 23),
            team_code="ARG",
            team_name="Argentina",
            rating=2147.0,
            global_rank=1,
        )
        s.add(snap)
        s.commit()
        loaded = s.scalar(select(RawEloSnapshot))
        assert loaded.team_code == "ARG"
        assert loaded.rating == 2147.0


def test_raw_team_asset_round_trip(sqlite_engine):
    with Session(sqlite_engine) as s:
        a = RawTeamAsset(
            team="Argentina",
            thesportsdb_id=133602,
            crest_url="https://example.test/argentina.png",
            kit_home_color="#75AADB",
            kit_away_color="#000080",
            stadium_name="Estadio Monumental",
            stadium_capacity=83214,
            stadium_city="Buenos Aires",
            stadium_country="Argentina",
            fetched_at=datetime.now(UTC),
        )
        s.add(a)
        s.commit()
        loaded = s.scalar(select(RawTeamAsset))
        assert loaded.team == "Argentina"
        assert loaded.stadium_capacity == 83214


def test_raw_squad_composite_pk_allows_history(sqlite_engine):
    """Same player across two snapshot_dates inserts two rows, not a conflict."""
    with Session(sqlite_engine) as s:
        for snap_date in (date(2026, 5, 1), date(2026, 6, 1)):
            s.add(
                RawSquad(
                    tournament="FIFA World Cup 2026",
                    team="Argentina",
                    player_name="Lionel Messi",
                    snapshot_date=snap_date,
                    shirt_number=10,
                    position="FWD",
                    birth_date=date(1987, 6, 24),
                    club="Inter Miami",
                    caps=190,
                    goals=109,
                    ingested_at=datetime.now(UTC),
                )
            )
        s.commit()
        rows = s.scalars(select(RawSquad)).all()
        assert len(rows) == 2
        assert {r.snapshot_date for r in rows} == {date(2026, 5, 1), date(2026, 6, 1)}


def test_raw_fifa_ranking_round_trip(sqlite_engine):
    with Session(sqlite_engine) as s:
        r = RawFifaRanking(
            ranking_date=date(2026, 4, 4),
            team="Argentina",
            rank=1,
            points=1886.16,
            previous_rank=1,
            ingested_at=datetime.now(UTC),
        )
        s.add(r)
        s.commit()
        loaded = s.scalar(select(RawFifaRanking))
        assert loaded.rank == 1
        assert loaded.points == 1886.16


def test_match_features_round_trip_with_json_provenance(sqlite_engine):
    with Session(sqlite_engine) as s:
        row = MatchFeatures(
            match_date=date(2026, 6, 11),
            home_team="Mexico",
            away_team="Senegal",
            elo_diff=-58.2,
            fifa_rank_diff=-5,
            xg_form_diff=0.34,
            rest_days_diff=2,
            squad_age_diff=0.7,
            is_neutral=0,
            is_host_home=1,
            is_host_away=0,
            poisson_exp_home_goals=1.46,
            poisson_exp_away_goals=1.10,
            poisson_p_home=0.47,
            poisson_p_draw=0.27,
            poisson_p_away=0.26,
            source_snapshots={
                "elo_snapshot_date": "2026-05-23",
                "model_version": "poisson_dc.v1",
            },
            built_at=datetime.now(UTC),
        )
        s.add(row)
        s.commit()
        loaded = s.scalar(select(MatchFeatures))
        assert loaded.is_host_home == 1
        assert loaded.source_snapshots == {
            "elo_snapshot_date": "2026-05-23",
            "model_version": "poisson_dc.v1",
        }
        assert abs((loaded.poisson_p_home + loaded.poisson_p_draw + loaded.poisson_p_away) - 1.0) < 1e-9


def test_raw_match_xg_composite_pk_allows_multiple_sources(sqlite_engine):
    """Same match, same team, two different xG sources insert as two rows."""
    with Session(sqlite_engine) as s:
        for src in ("statsbomb_open", "fbref"):
            s.add(
                RawMatchXg(
                    match_date=date(2022, 12, 18),
                    home_team="Argentina",
                    away_team="France",
                    team="Argentina",
                    source=src,
                    xg_for=1.9,
                    xg_against=2.7,
                    shots=21,
                    shots_on_target=8,
                    ingested_at=datetime.now(UTC),
                )
            )
        s.commit()
        rows = s.scalars(select(RawMatchXg)).all()
        assert len(rows) == 2
        assert {r.source for r in rows} == {"statsbomb_open", "fbref"}
