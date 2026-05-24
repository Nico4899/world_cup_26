"""Build the ``known_group_results`` map from the live-events table.

Phase 8 conditional Monte Carlo: the simulator accepts an optional dict
``{(home, away): (h_score, a_score)}``. This helper turns the FT_WHISTLE
rows the live poller has already written into that shape, by joining them
to the WC 2026 fixture set on football-data.org's match_id.
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from wc2026.db.models import RawLiveEvent
from wc2026.db.session import get_engine
from wc2026.ingest.live_events import EVENT_FT_WHISTLE


def known_group_results_from_live_events(
    match_id_to_fixture: dict[int, tuple[date, str, str]],
    *,
    engine: Engine | None = None,
) -> dict[tuple[str, str], tuple[int, int]]:
    """Read every FT_WHISTLE row, return the ``known_group_results`` dict.

    Empty mapping → empty result (the simulator then samples every fixture
    fresh, equivalent to a no-condition Monte Carlo). Unknown ``match_id``s
    (not in ``match_id_to_fixture``) are silently skipped so a stale
    football-data.org cache doesn't break the rerun.
    """
    if not match_id_to_fixture:
        return {}
    eng = engine or get_engine()
    with Session(eng, future=True) as session:
        rows = list(
            session.scalars(
                select(RawLiveEvent).where(RawLiveEvent.event_type == EVENT_FT_WHISTLE)
            )
        )
    out: dict[tuple[str, str], tuple[int, int]] = {}
    for r in rows:
        fixture = match_id_to_fixture.get(r.match_id)
        if fixture is None:
            continue
        _match_date, home_team, away_team = fixture
        out[(home_team, away_team)] = (int(r.home_score_after), int(r.away_score_after))
    return out


__all__ = ["known_group_results_from_live_events"]
