"""Full-stack integration: API + dashboard pages render against a live Docker stack.

Run with:
    docker compose up -d postgres api
    uv run pytest -m integration tests/integration/test_full_stack_smoke.py

If the API isn't reachable at WC2026_API_URL (default http://localhost:8000),
each test is skipped — we don't want CI noise when the stack isn't booted.

The test exists so that "fresh clone → docker compose up → it works" is
verified end-to-end, catching the seams between the API container, the
Streamlit pages, and the Pydantic schemas that unit tests can't.
"""

from __future__ import annotations

import os
import socket
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pytest

API_URL = os.environ.get("WC2026_API_URL", "http://localhost:8000")
DASHBOARD_DIR = Path(__file__).resolve().parent.parent.parent / "dashboard"
PAGES_DIR = DASHBOARD_DIR / "pages"
ENTRYPOINT = DASHBOARD_DIR / "streamlit_app.py"

pytestmark = pytest.mark.integration


def _api_reachable(timeout_s: float = 1.0) -> bool:
    """Quick TCP check: is something listening on the API's host:port?"""
    parsed = urlparse(API_URL)
    host = parsed.hostname or "localhost"
    port = parsed.port or 80
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _wait_for_api_ready(timeout_s: float = 60.0) -> dict:
    """Poll /health until model_fitted=true. Returns the final /health body."""
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = httpx.get(f"{API_URL}/health", timeout=5.0)
            if r.status_code == 200:
                body = r.json()
                if body.get("model_fitted"):
                    return body
        except httpx.HTTPError as exc:
            last_err = exc
        time.sleep(2.0)
    raise TimeoutError(f"API never became ready at {API_URL} ({last_err!r})")


@pytest.fixture(scope="module")
def api_ready() -> dict:
    if not _api_reachable():
        pytest.skip(
            f"API not reachable at {API_URL} — run `docker compose up -d postgres api` first"
        )
    return _wait_for_api_ready()


# --- API smoke -------------------------------------------------------------


def test_api_health_reports_fitted_model(api_ready: dict) -> None:
    assert api_ready["status"] == "ok"
    assert api_ready["model_fitted"] is True
    assert api_ready["model_teams_n"] > 100


def test_api_returns_72_wc_fixtures(api_ready: dict) -> None:
    _ = api_ready
    r = httpx.get(f"{API_URL}/api/v1/matches", timeout=10.0)
    assert r.status_code == 200
    matches = r.json()
    assert len(matches) == 72


def test_api_predictions_endpoint_returns_normalised_probabilities(api_ready: dict) -> None:
    _ = api_ready
    r = httpx.get(f"{API_URL}/api/v1/predictions/Argentina/France", timeout=10.0)
    assert r.status_code == 200
    out = r.json()
    s = out["outcome"]["home_win"] + out["outcome"]["draw"] + out["outcome"]["away_win"]
    assert abs(s - 1.0) < 1e-6


# --- Streamlit AppTest -----------------------------------------------------


def _streamlit_targets() -> list[Path]:
    """The entrypoint + every page under dashboard/pages/."""
    pages = sorted(PAGES_DIR.glob("*.py"))
    return [ENTRYPOINT, *pages]


@pytest.mark.parametrize("script", _streamlit_targets(), ids=lambda p: p.name)
def test_dashboard_page_renders_without_exception(script: Path, api_ready: dict) -> None:
    """Each Streamlit page must import + execute against the live API without raising."""
    _ = api_ready
    from streamlit.testing.v1 import AppTest  # local import keeps unit-test boot fast

    # AppTest reads WC2026_API_URL from the environment via os.environ; ensure it points
    # at the same API our smoke checks already confirmed.
    os.environ["WC2026_API_URL"] = API_URL

    at = AppTest.from_file(str(script), default_timeout=30.0)
    at.run()
    assert not at.exception, f"{script.name} raised: " + "; ".join(str(e) for e in at.exception)
