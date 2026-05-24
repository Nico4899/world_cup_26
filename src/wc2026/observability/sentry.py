"""Sentry SDK init — called from both the API lifespan and the scheduler entrypoint.

When ``SENTRY_DSN`` is unset, ``init_sentry`` is a silent no-op. We don't fail
the process just because monitoring isn't configured — that would defeat the
"degrade gracefully" promise the rest of the codebase keeps.

Environment variables we honor:
    SENTRY_DSN              required to enable Sentry; absent → no-op
    SENTRY_ENVIRONMENT      tag (default ``production`` when DSN set, else ``local``)
    SENTRY_TRACES_SAMPLE_RATE  float in [0, 1]; default 0.0 (errors only, no tracing)
    SENTRY_PROFILES_SAMPLE_RATE float in [0, 1]; default 0.0
    SENTRY_RELEASE          override; default is the package version from importlib.metadata
"""

from __future__ import annotations

import logging
import os
from importlib.metadata import PackageNotFoundError, version

import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

ENV_DSN = "SENTRY_DSN"
ENV_ENVIRONMENT = "SENTRY_ENVIRONMENT"
ENV_TRACES_SAMPLE_RATE = "SENTRY_TRACES_SAMPLE_RATE"
ENV_PROFILES_SAMPLE_RATE = "SENTRY_PROFILES_SAMPLE_RATE"
ENV_RELEASE = "SENTRY_RELEASE"

DEFAULT_PACKAGE = "wc2026-predictor"

logger = logging.getLogger(__name__)


def _resolve_release() -> str | None:
    """Best-effort: pick a release tag from env, then package version, then None."""
    env_release = os.environ.get(ENV_RELEASE)
    if env_release:
        return env_release
    try:
        return f"{DEFAULT_PACKAGE}@{version(DEFAULT_PACKAGE)}"
    except PackageNotFoundError:
        return None


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; falling back to %f", name, raw, default)
        return default


def init_sentry(*, service: str) -> bool:
    """Initialise Sentry. Returns True if Sentry was actually enabled.

    ``service`` is recorded as a tag so we can distinguish "api" vs "scheduler"
    events without filtering on stack-trace heuristics.

    Idempotent: safe to call multiple times in the same process — Sentry's
    own SDK deduplicates on the DSN.
    """
    dsn = os.environ.get(ENV_DSN)
    if not dsn:
        logger.debug("SENTRY_DSN unset; Sentry disabled (%s)", service)
        return False
    environment = os.environ.get(ENV_ENVIRONMENT) or "production"
    traces = _float_env(ENV_TRACES_SAMPLE_RATE, 0.0)
    profiles = _float_env(ENV_PROFILES_SAMPLE_RATE, 0.0)
    release = _resolve_release()

    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        release=release,
        traces_sample_rate=traces,
        profiles_sample_rate=profiles,
        # Capture ERROR-and-above as events; don't ship every INFO line.
        integrations=[
            LoggingIntegration(level=logging.INFO, event_level=logging.ERROR),
        ],
        send_default_pii=False,
    )
    sentry_sdk.set_tag("service", service)
    logger.info("Sentry initialised (service=%s, environment=%s)", service, environment)
    return True


__all__ = [
    "DEFAULT_PACKAGE",
    "ENV_DSN",
    "ENV_ENVIRONMENT",
    "ENV_PROFILES_SAMPLE_RATE",
    "ENV_RELEASE",
    "ENV_TRACES_SAMPLE_RATE",
    "init_sentry",
]
