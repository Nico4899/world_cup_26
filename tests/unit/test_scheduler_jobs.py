"""Unit tests for the APScheduler job module (no live scheduling)."""

from __future__ import annotations

from datetime import date

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from wc2026.scheduler import jobs as job_mod


def test_job_specs_have_ten_entries_at_distinct_slots():
    slots = {(s.hour, s.minute, s.day_of_week, s.day) for s in job_mod.JOB_SPECS}
    assert len(job_mod.JOB_SPECS) == 10
    assert len(slots) == 10, "expected ten distinct (hour, minute, day_of_week, day) slots"


def test_job_specs_use_expected_names_and_window():
    names = {s.name for s in job_mod.JOB_SPECS}
    assert names == {
        "db_backup",
        "kaggle_refresh",
        "elo_refresh",
        "football_data_org_refresh",
        "poisson_refit",
        "thesportsdb_refresh",
        "openfootball_refresh",
        "fifa_ranking_refresh",
        "football_data_co_uk_refresh",
        "fbref_refresh",
    }
    for spec in job_mod.JOB_SPECS:
        # 02:xx backup, 03:xx weekly metadata + odds, 04:xx ingest, 05:00 refit
        # (+ 05:30 weekly FBref), 06:00 monthly ranking.
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


def test_manual_only_specs_include_squads_and_statsbomb():
    names = {s.name for s in job_mod.MANUAL_ONLY_JOB_SPECS}
    assert names == {"wikipedia_squads_refresh", "statsbomb_refresh"}


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
