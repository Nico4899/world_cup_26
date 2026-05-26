"""Unit tests for the APScheduler job module (no live scheduling)."""

from __future__ import annotations

from datetime import date

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from wc2026.scheduler import jobs as job_mod


def test_job_specs_have_thirteen_entries_at_distinct_slots():
    slots = {(s.hour, s.minute, s.day_of_week, s.day) for s in job_mod.JOB_SPECS}
    assert len(job_mod.JOB_SPECS) == 13
    assert len(slots) == 13, "expected thirteen distinct (hour, minute, day_of_week, day) slots"


def test_job_specs_use_expected_names_and_window():
    names = {s.name for s in job_mod.JOB_SPECS}
    assert names == {
        "db_backup",
        "kaggle_refresh",
        "elo_refresh",
        "football_data_org_refresh",
        "poisson_refit",
        "features_rebuild",
        "thesportsdb_refresh",
        "openfootball_refresh",
        "fifa_ranking_refresh",
        "football_data_co_uk_refresh",
        "fbref_refresh",
        "xgb_refit",
        "climate_refresh",
    }
    for spec in job_mod.JOB_SPECS:
        # 02:xx backup, 03:xx weekly metadata + odds, 04:xx ingest, 05:00 refit,
        # 05:15 daily feature rebuild, 05:30 weekly FBref, 05:45 weekly XGB refit,
        # 06:00 monthly ranking.
        assert spec.hour in (2, 3, 4, 5, 6)
        assert spec.minute in {0, 15, 30, 45}


def test_weekly_jobs_are_marked_with_day_of_week():
    by_name = {s.name: s for s in job_mod.JOB_SPECS}
    assert by_name["thesportsdb_refresh"].day_of_week == "sun"
    assert by_name["openfootball_refresh"].day_of_week == "sun"
    # Daily jobs leave both fields as None.
    assert by_name["db_backup"].day_of_week is None
    assert by_name["db_backup"].day is None


def test_monthly_fifa_ranking_job_pins_day_one():
    by_name = {s.name: s for s in job_mod.JOB_SPECS}
    assert by_name["fifa_ranking_refresh"].day == 1
    assert by_name["fifa_ranking_refresh"].day_of_week is None


def test_manual_only_specs_include_squads_statsbomb_and_transfermarkt():
    names = {s.name for s in job_mod.MANUAL_ONLY_JOB_SPECS}
    assert names == {
        "wikipedia_squads_refresh",
        "statsbomb_refresh",
        "transfermarkt_refresh",
    }


def test_job_registry_covers_cron_and_manual_jobs():
    expected = {s.name for s in job_mod.JOB_SPECS} | {s.name for s in job_mod.MANUAL_ONLY_JOB_SPECS}
    assert set(job_mod.JOB_REGISTRY) == expected


def test_register_jobs_attaches_all_cron_triggers():
    scheduler = BackgroundScheduler(timezone="UTC")
    job_mod.register_jobs(scheduler)
    assert len(scheduler.get_jobs()) == len(job_mod.JOB_SPECS)
    for j in scheduler.get_jobs():
        assert isinstance(j.trigger, CronTrigger)
        assert str(j.trigger.timezone) == "UTC"


def test_build_scheduler_returns_blocking_scheduler_with_jobs():
    from apscheduler.schedulers.blocking import BlockingScheduler

    scheduler = job_mod.build_scheduler()
    assert isinstance(scheduler, BlockingScheduler)
    assert len(scheduler.get_jobs()) == len(job_mod.JOB_SPECS)
    ids = {j.id for j in scheduler.get_jobs()}
    assert ids == {s.name for s in job_mod.JOB_SPECS}


def test_register_jobs_uses_expected_cron_fields():
    scheduler = BackgroundScheduler(timezone="UTC")
    job_mod.register_jobs(scheduler)
    by_name = {j.id: j for j in scheduler.get_jobs()}
    for spec in job_mod.JOB_SPECS:
        trig = by_name[spec.name].trigger
        fields = {f.name: str(f) for f in trig.fields}
        assert fields["hour"] == str(spec.hour)
        assert fields["minute"] == str(spec.minute)
        if spec.day_of_week is not None:
            assert fields["day_of_week"] == spec.day_of_week
        if spec.day is not None:
            assert fields["day"] == str(spec.day)


def test_wrap_with_tracking_records_ok_when_func_succeeds(monkeypatch):
    calls: list[tuple[str, str, str | None]] = []

    def fake_record(name, started, status, error_text):
        calls.append((name, status, error_text))

    monkeypatch.setattr(job_mod, "_record_job_run", fake_record)

    spec = job_mod.JobSpec(name="t_ok", hour=0, minute=0, func=lambda: None)
    job_mod._wrap_with_tracking(spec)()
    assert calls == [("t_ok", "ok", None)]


def test_wrap_with_tracking_records_error_when_func_raises(monkeypatch):
    calls: list[tuple[str, str, str | None]] = []

    def fake_record(name, started, status, error_text):
        calls.append((name, status, error_text))

    monkeypatch.setattr(job_mod, "_record_job_run", fake_record)

    def boom() -> None:
        raise RuntimeError("kaboom")

    spec = job_mod.JobSpec(name="t_err", hour=0, minute=0, func=boom)
    job_mod._wrap_with_tracking(spec)()
    assert len(calls) == 1
    name, status, err = calls[0]
    assert name == "t_err"
    assert status == "error"
    assert "kaboom" in err


def test_features_rebuild_job_no_ops_without_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("WC2026_DATABASE_URL", raising=False)
    # Patch the build script in scripts.build_features so a stray attempt to
    # call it would be visible.
    import scripts.build_features as bf

    called = {"n": 0}

    def fake_build(**_):
        called["n"] += 1
        return 0

    monkeypatch.setattr(bf, "build_and_persist_features", fake_build)
    job_mod._job_features_rebuild()
    assert called["n"] == 0


def test_features_rebuild_job_calls_build_when_database_url_set(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x:y@z/db")
    import scripts.build_features as bf
    import scripts.persist_wc2026_predictions as pp

    called = {"features": 0, "predictions": 0}

    def fake_build(**_):
        called["features"] += 1
        return 72

    def fake_persist(**_):
        called["predictions"] += 1
        return 72

    monkeypatch.setattr(bf, "build_and_persist_features", fake_build)
    monkeypatch.setattr(pp, "persist_daily_snapshot", fake_persist)
    job_mod._job_features_rebuild()
    # Both downstream pieces run on a successful pass.
    assert called == {"features": 1, "predictions": 1}


def test_football_data_job_no_ops_without_api_key(monkeypatch):
    monkeypatch.delenv("FOOTBALL_DATA_ORG_KEY", raising=False)

    called = {"fetch": 0}

    def fake_fetch(*args, **kwargs):
        called["fetch"] += 1

    monkeypatch.setattr(job_mod, "fetch_competition_matches", fake_fetch)

    job_mod._job_football_data_refresh()
    assert called["fetch"] == 0


# --- db_backup --------------------------------------------------------------


def test_db_backup_no_ops_without_database_url(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    out = job_mod._job_db_backup(backup_dir=tmp_path)
    assert out is None
    # Backup dir should NOT be auto-created when we early-returned.
    # (mkdir only happens after the env check.)
    assert not any(tmp_path.iterdir())


def test_db_backup_no_ops_when_pg_dump_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x:y@z/db")
    monkeypatch.setattr(job_mod.shutil, "which", lambda _: None)
    out = job_mod._job_db_backup(backup_dir=tmp_path)
    assert out is None


def test_db_backup_writes_gzipped_dump(monkeypatch, tmp_path):
    """Mock pg_dump to return a fixed payload; verify we gzip and write it."""
    import gzip as _gzip
    import subprocess as _subprocess

    monkeypatch.setenv("DATABASE_URL", "postgresql://x:y@z/db")
    monkeypatch.setattr(job_mod.shutil, "which", lambda _: "/usr/bin/pg_dump")

    fake_payload = b"-- mocked pg_dump output\nCREATE TABLE foo (id int);\n"

    class FakeResult:
        stdout = fake_payload
        stderr = b""

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeResult()

    monkeypatch.setattr(job_mod.subprocess, "run", fake_run)

    out = job_mod._job_db_backup(backup_dir=tmp_path)
    assert out is not None
    assert out.exists()
    assert out.name.startswith("wc2026-") and out.name.endswith(".sql.gz")
    # pg_dump command shape
    assert captured["cmd"][0] == "pg_dump"
    assert "--no-owner" in captured["cmd"]
    assert captured["cmd"][-1] == "postgresql://x:y@z/db"
    # Round-trip the gzip
    assert _gzip.decompress(out.read_bytes()) == fake_payload
    # Don't leak the unused import
    _ = _subprocess


def test_db_backup_prune_removes_files_older_than_retention(tmp_path):
    import os
    from datetime import UTC, datetime, timedelta

    # Three synthetic backups with mtimes at 20, 7, and 1 days ago.
    paths = []
    for days_ago in (20, 7, 1):
        p = tmp_path / f"wc2026-2026-05-{days_ago:02d}.sql.gz"
        p.write_bytes(b"x")
        ts = (datetime.now(UTC) - timedelta(days=days_ago)).timestamp()
        os.utime(p, (ts, ts))
        paths.append(p)
    # An unrelated file in the same dir should NOT be pruned.
    other = tmp_path / "do-not-touch.txt"
    other.write_text("keep me")
    os.utime(other, (paths[0].stat().st_mtime, paths[0].stat().st_mtime))

    removed = job_mod._prune_backups(tmp_path, retention_days=14)
    assert removed == 1  # only the 20-days-ago file
    assert not paths[0].exists()
    assert paths[1].exists() and paths[2].exists()
    assert other.exists()


def test_db_backup_prune_returns_zero_when_dir_missing(tmp_path):
    assert job_mod._prune_backups(tmp_path / "does-not-exist") == 0


def test_db_backup_calls_s3_upload_after_writing_local_dump(monkeypatch, tmp_path):
    """Phase 10: after pg_dump + local prune, the job invokes upload_backup so
    the off-site copy hits S3/R2. We don't require it to succeed (the upload
    helper is itself env-gated)."""
    import subprocess as _subprocess

    monkeypatch.setenv("DATABASE_URL", "postgresql://x:y@z/db")
    monkeypatch.setattr(job_mod.shutil, "which", lambda _: "/usr/bin/pg_dump")

    class FakeResult:
        stdout = b"-- mock"
        stderr = b""

    def fake_run(*args, **kwargs):
        _ = args, kwargs
        return FakeResult()

    monkeypatch.setattr(job_mod.subprocess, "run", fake_run)

    from wc2026.observability import s3_upload

    called: dict[str, int] = {"upload": 0}

    def fake_upload(local_path, **_kw):
        called["upload"] += 1

    monkeypatch.setattr(s3_upload, "upload_backup", fake_upload)

    out = job_mod._job_db_backup(backup_dir=tmp_path)
    assert out is not None
    assert called["upload"] == 1
    # Don't leak the unused import
    _ = _subprocess


def test_statsbomb_refresh_chains_live_win_prob_refit(monkeypatch, tmp_path):
    """After the xG shot model refit, statsbomb_refresh also calls
    fit_live_win_prob so both artefacts land in one operator click."""
    import pandas as pd
    import scripts.fit_live_win_prob as live_wp_module

    import wc2026.ingest.statsbomb_open as sb
    import wc2026.models.xg_shot_model as xg

    # 1 fake "tournament path" so `paths` is truthy
    monkeypatch.setattr(job_mod, "fetch_all_tournament_shots", lambda: [tmp_path / "fake"])
    monkeypatch.setattr(sb, "load_shots_corpus", lambda: pd.DataFrame({"x": [1, 2, 3]}))

    called = {"xg": 0, "live_wp": 0}

    def fake_xg_fit(_corpus, **_kw):
        called["xg"] += 1

    def fake_live_wp_fit(**_kw):
        called["live_wp"] += 1
        return tmp_path / "live_win_prob.json"

    monkeypatch.setattr(xg, "fit_and_save", fake_xg_fit)
    monkeypatch.setattr(live_wp_module, "fit_and_save", fake_live_wp_fit)
    job_mod._job_statsbomb_refresh()
    assert called == {"xg": 1, "live_wp": 1}


def test_statsbomb_refresh_swallows_live_win_prob_empty_corpus(monkeypatch, tmp_path):
    """If the live-win-prob fit raises ValueError (no rows), the xG refit
    still succeeds — neither pulls the other down."""
    import pandas as pd
    import scripts.fit_live_win_prob as live_wp_module

    import wc2026.ingest.statsbomb_open as sb
    import wc2026.models.xg_shot_model as xg

    monkeypatch.setattr(job_mod, "fetch_all_tournament_shots", lambda: [tmp_path / "fake"])
    monkeypatch.setattr(sb, "load_shots_corpus", lambda: pd.DataFrame({"x": [1]}))

    called = {"xg": 0}

    def fake_xg_fit(_corpus, **_kw):
        called["xg"] += 1

    def fake_live_wp_fit(**_kw):
        raise ValueError("no rows — populate the StatsBomb corpus first")

    monkeypatch.setattr(xg, "fit_and_save", fake_xg_fit)
    monkeypatch.setattr(live_wp_module, "fit_and_save", fake_live_wp_fit)
    # Must not propagate — the warning path is the explicit no-op.
    job_mod._job_statsbomb_refresh()
    assert called["xg"] == 1


def test_db_backup_swallows_s3_upload_failure(monkeypatch, tmp_path):
    """An S3 upload failure must NOT take down the local backup job."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://x:y@z/db")
    monkeypatch.setattr(job_mod.shutil, "which", lambda _: "/usr/bin/pg_dump")

    class FakeResult:
        stdout = b"-- mock"
        stderr = b""

    monkeypatch.setattr(job_mod.subprocess, "run", lambda *a, **k: FakeResult())

    from wc2026.observability import s3_upload

    def boom(*_a, **_kw):
        raise RuntimeError("network is on fire")

    monkeypatch.setattr(s3_upload, "upload_backup", boom)

    out = job_mod._job_db_backup(backup_dir=tmp_path)
    assert out is not None  # the local backup still landed
    assert out.exists()


# --- tournament-window standings cache warm --------------------------------


def test_is_tournament_window_recognises_window_bounds():
    assert not job_mod.is_tournament_window(date(2026, 6, 10))
    assert job_mod.is_tournament_window(date(2026, 6, 11))  # opener
    assert job_mod.is_tournament_window(date(2026, 6, 25))
    assert job_mod.is_tournament_window(date(2026, 7, 19))  # final day
    assert not job_mod.is_tournament_window(date(2026, 7, 20))


def test_register_jobs_skips_standings_warm_outside_tournament_window():
    scheduler = BackgroundScheduler(timezone="UTC")
    job_mod.register_jobs(scheduler, today=date(2026, 5, 23))
    ids = {j.id for j in scheduler.get_jobs()}
    assert "standings_cache_warm" not in ids
    assert "monte_carlo_rerun" not in ids
    # All cron jobs still registered.
    assert ids == {s.name for s in job_mod.JOB_SPECS}


def test_register_jobs_adds_standings_warm_inside_tournament_window():
    scheduler = BackgroundScheduler(timezone="UTC")
    job_mod.register_jobs(scheduler, today=date(2026, 6, 25))
    by_id = {j.id: j for j in scheduler.get_jobs()}
    assert "standings_cache_warm" in by_id
    trig = by_id["standings_cache_warm"].trigger
    assert isinstance(trig, IntervalTrigger)
    # Interval should match the documented STANDINGS_WARM_INTERVAL_MINUTES.
    expected = job_mod.STANDINGS_WARM_INTERVAL_MINUTES * 60
    assert int(trig.interval.total_seconds()) == expected


def test_register_jobs_adds_monte_carlo_rerun_inside_tournament_window():
    scheduler = BackgroundScheduler(timezone="UTC")
    job_mod.register_jobs(scheduler, today=date(2026, 6, 25))
    by_id = {j.id: j for j in scheduler.get_jobs()}
    assert "monte_carlo_rerun" in by_id
    trig = by_id["monte_carlo_rerun"].trigger
    assert isinstance(trig, IntervalTrigger)
    expected = job_mod.MONTE_CARLO_RERUN_INTERVAL_MINUTES * 60
    assert int(trig.interval.total_seconds()) == expected


def test_monte_carlo_rerun_job_no_ops_outside_window(monkeypatch):
    """Even if directly invoked outside the window, the job mustn't call the rerun."""
    import scripts.rerun_monte_carlo as mcr

    monkeypatch.setattr(job_mod, "WC_TOURNAMENT_START", date(2099, 1, 1))

    called = {"n": 0}

    def fake_rerun(**_):
        called["n"] += 1

    monkeypatch.setattr(mcr, "rerun_and_persist", fake_rerun)
    job_mod._job_monte_carlo_rerun()
    assert called["n"] == 0


def test_monte_carlo_rerun_job_invokes_rerun_inside_window(monkeypatch):
    import scripts.rerun_monte_carlo as mcr

    monkeypatch.setattr(job_mod, "WC_TOURNAMENT_START", date(1900, 1, 1))
    monkeypatch.setattr(job_mod, "WC_TOURNAMENT_END", date(2999, 12, 31))

    called = {"n": 0}

    def fake_rerun(**_):
        called["n"] += 1
        return 42

    monkeypatch.setattr(mcr, "rerun_and_persist", fake_rerun)
    job_mod._job_monte_carlo_rerun()
    assert called["n"] == 1


# --- live_events_poll (Phase 6 production poller) --------------------------


def test_register_jobs_adds_live_events_poll_inside_tournament_window():
    scheduler = BackgroundScheduler(timezone="UTC")
    job_mod.register_jobs(scheduler, today=date(2026, 6, 25))
    by_id = {j.id: j for j in scheduler.get_jobs()}
    assert "live_events_poll" in by_id
    trig = by_id["live_events_poll"].trigger
    assert isinstance(trig, IntervalTrigger)
    assert int(trig.interval.total_seconds()) == job_mod.LIVE_EVENTS_POLL_INTERVAL_SECONDS


def test_register_jobs_skips_live_events_poll_outside_tournament_window():
    scheduler = BackgroundScheduler(timezone="UTC")
    job_mod.register_jobs(scheduler, today=date(2026, 5, 23))
    assert "live_events_poll" not in {j.id for j in scheduler.get_jobs()}


def test_live_events_poll_no_ops_outside_window(monkeypatch):
    monkeypatch.setattr(job_mod, "WC_TOURNAMENT_START", date(2099, 1, 1))
    monkeypatch.setenv("FOOTBALL_DATA_ORG_KEY", "x")
    monkeypatch.setenv("DATABASE_URL", "x")
    import wc2026.ingest.football_data_org as fdo
    import wc2026.ingest.live_events as live_ev

    def boom_fetch(*_, **__):
        raise AssertionError("must not fetch outside window")

    def boom_poll(*_, **__):
        raise AssertionError("must not poll outside window")

    monkeypatch.setattr(fdo, "fetch_competition_matches", boom_fetch)
    monkeypatch.setattr(live_ev, "poll_live_match", boom_poll)
    job_mod._job_live_events_poll()  # no assertion fires → pass


def test_live_events_poll_no_ops_without_football_data_org_key(monkeypatch):
    monkeypatch.setattr(job_mod, "WC_TOURNAMENT_START", date(1900, 1, 1))
    monkeypatch.setattr(job_mod, "WC_TOURNAMENT_END", date(2999, 12, 31))
    monkeypatch.delenv("FOOTBALL_DATA_ORG_KEY", raising=False)
    monkeypatch.setenv("DATABASE_URL", "x")
    import wc2026.ingest.football_data_org as fdo

    def boom_fetch(*_, **__):
        raise AssertionError("must not fetch without API key")

    monkeypatch.setattr(fdo, "fetch_competition_matches", boom_fetch)
    job_mod._job_live_events_poll()


def test_live_events_poll_no_ops_without_database_url(monkeypatch):
    monkeypatch.setattr(job_mod, "WC_TOURNAMENT_START", date(1900, 1, 1))
    monkeypatch.setattr(job_mod, "WC_TOURNAMENT_END", date(2999, 12, 31))
    monkeypatch.setenv("FOOTBALL_DATA_ORG_KEY", "x")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("WC2026_DATABASE_URL", raising=False)
    import wc2026.ingest.football_data_org as fdo

    def boom_fetch(*_, **__):
        raise AssertionError("must not fetch without DB URL")

    monkeypatch.setattr(fdo, "fetch_competition_matches", boom_fetch)
    job_mod._job_live_events_poll()


def test_live_events_poll_filters_to_today_and_polls(monkeypatch):
    """Happy path: fixture list has 3 matches — 1 today IN_PLAY, 1 today SCHEDULED,
    1 yesterday FINISHED. We expect 2 polls (the IN_PLAY today + the FINISHED yesterday)."""
    import datetime as _dt

    import pandas as _pd

    import wc2026.ingest.football_data_org as fdo
    import wc2026.ingest.live_events as live_ev

    today = _dt.date(2026, 6, 25)
    monkeypatch.setattr(job_mod, "WC_TOURNAMENT_START", _dt.date(1900, 1, 1))
    monkeypatch.setattr(job_mod, "WC_TOURNAMENT_END", _dt.date(2999, 12, 31))

    class _FrozenDatetime:
        @classmethod
        def now(cls, _tz=None):
            return _dt.datetime(2026, 6, 25, 14, 0, tzinfo=_dt.UTC)

    monkeypatch.setattr(job_mod, "datetime", _FrozenDatetime)
    monkeypatch.setenv("FOOTBALL_DATA_ORG_KEY", "x")
    monkeypatch.setenv("DATABASE_URL", "x")

    df = _pd.DataFrame(
        {
            "match_id": [101, 102, 103],
            "status": ["IN_PLAY", "SCHEDULED", "FINISHED"],
            "utc_date": _pd.to_datetime(
                [
                    f"{today.isoformat()} 13:00:00+00:00",
                    f"{today.isoformat()} 19:00:00+00:00",
                    f"{(today - _dt.timedelta(days=1)).isoformat()} 21:00:00+00:00",
                ],
                utc=True,
            ),
        }
    )

    monkeypatch.setattr(fdo, "fetch_competition_matches", lambda *_a, **_k: df)

    polled: list[int] = []

    def fake_poll(match_id, **_kw):
        polled.append(int(match_id))

    monkeypatch.setattr(live_ev, "poll_live_match", fake_poll)
    job_mod._job_live_events_poll()
    assert sorted(polled) == [101, 103]  # SCHEDULED 102 is filtered out


def test_warm_standings_cache_no_ops_outside_window(monkeypatch):
    """Even if directly invoked outside the tournament window, the job should
    not hit the network (defensive guard against mis-registration)."""
    import httpx as _httpx

    called = {"get": 0}

    def fake_get(*args, **kwargs):
        called["get"] += 1
        raise AssertionError("must not call httpx.get outside the window")

    monkeypatch.setattr(_httpx, "get", fake_get)
    # Frozen "today" via a temporary monkeypatch of datetime in the module.
    monkeypatch.setattr(
        job_mod,
        "WC_TOURNAMENT_START",
        date(2099, 1, 1),  # window in the far future → today is "outside"
    )
    job_mod._job_warm_standings_cache(api_url="http://fake")
    assert called["get"] == 0


def test_warm_standings_cache_calls_api_inside_window(monkeypatch):
    import httpx as _httpx

    monkeypatch.setattr(job_mod, "WC_TOURNAMENT_START", date(1900, 1, 1))
    monkeypatch.setattr(job_mod, "WC_TOURNAMENT_END", date(2999, 12, 31))

    captured = {}

    class FakeResp:
        def raise_for_status(self) -> None:
            return None

    def fake_get(url, *, timeout):
        captured["url"] = url
        captured["timeout"] = timeout
        return FakeResp()

    monkeypatch.setattr(_httpx, "get", fake_get)
    job_mod._job_warm_standings_cache(api_url="http://api.example:8000/")
    assert captured["url"] == "http://api.example:8000/api/v1/tournament/standings"


def test_warm_standings_cache_swallows_http_errors(monkeypatch, caplog):
    import httpx as _httpx

    monkeypatch.setattr(job_mod, "WC_TOURNAMENT_START", date(1900, 1, 1))
    monkeypatch.setattr(job_mod, "WC_TOURNAMENT_END", date(2999, 12, 31))

    def fake_get(*args, **kwargs):
        raise _httpx.ConnectError("boom")

    monkeypatch.setattr(_httpx, "get", fake_get)
    # Must not raise.
    job_mod._job_warm_standings_cache(api_url="http://fake")


def test_climate_refresh_writes_one_json_per_upcoming_fixture(monkeypatch, tmp_path):
    """Happy path: stubbed Open-Meteo + scheduled fixtures → 1 JSON per pair."""
    import json
    from datetime import UTC, datetime, timedelta

    import pandas as pd

    from wc2026.features import venue as venue_mod
    from wc2026.ingest import kaggle_intl

    today = datetime.now(UTC).date()
    scheduled = pd.DataFrame(
        {
            "date": [pd.Timestamp(today + timedelta(days=1)), pd.Timestamp(today + timedelta(days=2))],
            "home_team": ["Mexico", "Argentina"],
            "away_team": ["Saudi Arabia", "Poland"],
            "city": ["Mexico City", "Miami"],
        }
    )
    monkeypatch.setattr(kaggle_intl, "load_scheduled", lambda: scheduled)
    monkeypatch.setattr(venue_mod, "_wet_bulb_from_open_meteo", lambda *_a, **_kw: 14.0)
    monkeypatch.setattr(job_mod, "CLIMATE_RAW_DIR", tmp_path)

    job_mod._job_climate_refresh()

    files = sorted(tmp_path.glob("*.json"))
    assert len(files) == 2
    payload = json.loads(files[0].read_text())
    assert payload["wet_bulb_c"] == 14.0
    assert "city" in payload and "match_date" in payload and "fetched_at" in payload


def test_climate_refresh_skips_when_open_meteo_returns_none(monkeypatch, tmp_path):
    """API failure path: no files written, no exception raised."""
    from datetime import UTC, datetime, timedelta

    import pandas as pd

    from wc2026.features import venue as venue_mod
    from wc2026.ingest import kaggle_intl

    today = datetime.now(UTC).date()
    scheduled = pd.DataFrame(
        {
            "date": [pd.Timestamp(today + timedelta(days=1))],
            "home_team": ["Mexico"],
            "away_team": ["Saudi Arabia"],
            "city": ["Mexico City"],
        }
    )
    monkeypatch.setattr(kaggle_intl, "load_scheduled", lambda: scheduled)
    monkeypatch.setattr(venue_mod, "_wet_bulb_from_open_meteo", lambda *_a, **_kw: None)
    monkeypatch.setattr(job_mod, "CLIMATE_RAW_DIR", tmp_path)

    job_mod._job_climate_refresh()  # must not raise
    assert list(tmp_path.glob("*.json")) == []


def test_climate_refresh_no_ops_without_scheduled_csv(monkeypatch, tmp_path):
    """If the Kaggle scheduled CSV is missing, the job logs + returns cleanly."""
    from wc2026.ingest import kaggle_intl

    def boom() -> object:
        raise FileNotFoundError("no scheduled CSV")

    monkeypatch.setattr(kaggle_intl, "load_scheduled", boom)
    monkeypatch.setattr(job_mod, "CLIMATE_RAW_DIR", tmp_path)
    job_mod._job_climate_refresh()  # must not raise
    assert list(tmp_path.glob("*.json")) == []
