"""Integration test: alembic upgrade head + round-trip insert against docker Postgres.

Run with:
    docker compose up -d postgres
    uv run pytest -m integration tests/integration/test_db_roundtrip.py

If WC2026_DATABASE_URL is not set, the test falls back to the docker-compose URL
on port 55432. The test creates a *separate* schema, runs migrations against it,
inserts a row, and drops the schema, so it never collides with real data.
"""

from __future__ import annotations

import os
import socket
import uuid
from datetime import UTC, date, datetime

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from wc2026.db.models import RawMatch
from wc2026.db.session import DEFAULT_LOCAL_URL


def _can_connect(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.integration


@pytest.fixture
def pg_url() -> str:
    url = os.environ.get("WC2026_DATABASE_URL", DEFAULT_LOCAL_URL)
    if "@" not in url or not _can_connect("localhost", 55432):
        pytest.skip(
            "Postgres not reachable on localhost:55432 — run `docker compose up -d postgres`"
        )
    return url


def test_alembic_upgrade_then_insert_roundtrip(pg_url):
    schema = f"wc2026_test_{uuid.uuid4().hex[:8]}"
    engine = create_engine(pg_url, future=True)
    try:
        with engine.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA "{schema}"'))
            conn.execute(text(f'SET search_path TO "{schema}"'))

        os.environ["WC2026_DATABASE_URL"] = (
            pg_url + ("&" if "?" in pg_url else "?") + f"options=-csearch_path%3D{schema}"
        )
        cfg = Config("alembic.ini")
        command.upgrade(cfg, "head")

        with Session(engine) as s:
            s.execute(text(f'SET search_path TO "{schema}"'))
            m = RawMatch(
                date=date(2026, 6, 11),
                home_team="Mexico",
                away_team="Senegal",
                tournament="FIFA World Cup",
                neutral=False,
                source="integration_test",
                ingested_at=datetime.now(UTC),
            )
            s.add(m)
            s.commit()
            count = s.execute(text("SELECT count(*) FROM raw_matches")).scalar_one()
            assert count == 1
    finally:
        with engine.begin() as conn:
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        engine.dispose()
        os.environ.pop("WC2026_DATABASE_URL", None)
