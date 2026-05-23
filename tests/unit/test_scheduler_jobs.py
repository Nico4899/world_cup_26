"""Unit tests for the APScheduler job module (no live scheduling)."""

from __future__ import annotations

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from wc2026.scheduler import jobs as job_mod


def test_job_specs_have_three_entries_at_distinct_times():
    times = {(s.hour, s.minute) for s in job_mod.JOB_SPECS}
    assert len(job_mod.JOB_SPECS) == 3
    assert len(times) == 3, "expected three distinct (hour, minute) slots"


def test_job_specs_use_expected_names_and_window():
    names = {s.name for s in job_mod.JOB_SPECS}
    assert names == {"kaggle_refresh", "elo_refresh", "football_data_org_refresh"}
    for spec in job_mod.JOB_SPECS:
        assert spec.hour == 4, "all daily jobs run at 04:xx UTC"
        assert spec.minute in {0, 15, 30}


def test_register_jobs_attaches_three_cron_triggers():
    scheduler = BackgroundScheduler(timezone="UTC")
    job_mod.register_jobs(scheduler)
    assert len(scheduler.get_jobs()) == 3
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
