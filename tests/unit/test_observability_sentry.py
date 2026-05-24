"""Unit tests for the Sentry init helper.

We don't actually want to ship events to Sentry from CI, so every test
patches ``sentry_sdk.init`` and asserts on the kwargs we pass. The DSN-unset
branch must return False without touching the SDK at all.
"""

from __future__ import annotations

from typing import Any

from wc2026.observability import sentry as sentry_mod


def test_init_returns_false_when_dsn_is_unset(monkeypatch) -> None:
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    called: dict[str, Any] = {}

    def fake_init(**kwargs):
        called["kwargs"] = kwargs

    monkeypatch.setattr(sentry_mod.sentry_sdk, "init", fake_init)
    enabled = sentry_mod.init_sentry(service="api")
    assert enabled is False
    assert called == {}  # sentry_sdk.init must NOT be called


def test_init_returns_true_when_dsn_is_set(monkeypatch) -> None:
    monkeypatch.setenv("SENTRY_DSN", "https://abc@sentry.test/1")
    monkeypatch.delenv("SENTRY_ENVIRONMENT", raising=False)
    tags: dict[str, str] = {}
    init_kwargs: dict[str, Any] = {}

    def fake_init(**kwargs):
        init_kwargs.update(kwargs)

    def fake_set_tag(key: str, value: str) -> None:
        tags[key] = value

    monkeypatch.setattr(sentry_mod.sentry_sdk, "init", fake_init)
    monkeypatch.setattr(sentry_mod.sentry_sdk, "set_tag", fake_set_tag)
    enabled = sentry_mod.init_sentry(service="scheduler")
    assert enabled is True
    assert init_kwargs["dsn"] == "https://abc@sentry.test/1"
    assert init_kwargs["environment"] == "production"
    # Default sample rates: 0.0 (errors only).
    assert init_kwargs["traces_sample_rate"] == 0.0
    assert init_kwargs["profiles_sample_rate"] == 0.0
    assert init_kwargs["send_default_pii"] is False
    assert tags == {"service": "scheduler"}


def test_init_honors_environment_override(monkeypatch) -> None:
    monkeypatch.setenv("SENTRY_DSN", "https://abc@sentry.test/1")
    monkeypatch.setenv("SENTRY_ENVIRONMENT", "staging")
    init_kwargs: dict[str, Any] = {}

    monkeypatch.setattr(sentry_mod.sentry_sdk, "init", lambda **k: init_kwargs.update(k))
    monkeypatch.setattr(sentry_mod.sentry_sdk, "set_tag", lambda *_: None)
    sentry_mod.init_sentry(service="api")
    assert init_kwargs["environment"] == "staging"


def test_init_honors_traces_sample_rate(monkeypatch) -> None:
    monkeypatch.setenv("SENTRY_DSN", "https://abc@sentry.test/1")
    monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "0.25")
    init_kwargs: dict[str, Any] = {}

    monkeypatch.setattr(sentry_mod.sentry_sdk, "init", lambda **k: init_kwargs.update(k))
    monkeypatch.setattr(sentry_mod.sentry_sdk, "set_tag", lambda *_: None)
    sentry_mod.init_sentry(service="api")
    assert init_kwargs["traces_sample_rate"] == 0.25


def test_init_falls_back_to_default_on_bad_sample_rate(monkeypatch) -> None:
    monkeypatch.setenv("SENTRY_DSN", "https://abc@sentry.test/1")
    monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "not a float")
    init_kwargs: dict[str, Any] = {}

    monkeypatch.setattr(sentry_mod.sentry_sdk, "init", lambda **k: init_kwargs.update(k))
    monkeypatch.setattr(sentry_mod.sentry_sdk, "set_tag", lambda *_: None)
    sentry_mod.init_sentry(service="api")
    # The bad value is logged + default (0.0) is used.
    assert init_kwargs["traces_sample_rate"] == 0.0


def test_init_release_uses_env_override_when_present(monkeypatch) -> None:
    monkeypatch.setenv("SENTRY_DSN", "https://abc@sentry.test/1")
    monkeypatch.setenv("SENTRY_RELEASE", "wc2026-predictor@1.2.3-rc1")
    init_kwargs: dict[str, Any] = {}

    monkeypatch.setattr(sentry_mod.sentry_sdk, "init", lambda **k: init_kwargs.update(k))
    monkeypatch.setattr(sentry_mod.sentry_sdk, "set_tag", lambda *_: None)
    sentry_mod.init_sentry(service="api")
    assert init_kwargs["release"] == "wc2026-predictor@1.2.3-rc1"


def test_init_release_falls_back_to_package_version(monkeypatch) -> None:
    monkeypatch.setenv("SENTRY_DSN", "https://abc@sentry.test/1")
    monkeypatch.delenv("SENTRY_RELEASE", raising=False)
    init_kwargs: dict[str, Any] = {}

    monkeypatch.setattr(sentry_mod.sentry_sdk, "init", lambda **k: init_kwargs.update(k))
    monkeypatch.setattr(sentry_mod.sentry_sdk, "set_tag", lambda *_: None)
    sentry_mod.init_sentry(service="api")
    # Either populates from importlib.metadata or returns None — both fine.
    release = init_kwargs["release"]
    assert release is None or release.startswith("wc2026-predictor@")


def test_init_sets_service_tag(monkeypatch) -> None:
    monkeypatch.setenv("SENTRY_DSN", "https://abc@sentry.test/1")
    tags: dict[str, str] = {}

    def _capture_tag(k: str, v: str) -> None:
        tags.setdefault(k, v)

    def _noop_init(**_kw) -> None:
        return

    monkeypatch.setattr(sentry_mod.sentry_sdk, "init", _noop_init)
    monkeypatch.setattr(sentry_mod.sentry_sdk, "set_tag", _capture_tag)
    sentry_mod.init_sentry(service="custom-worker")
    assert tags == {"service": "custom-worker"}
