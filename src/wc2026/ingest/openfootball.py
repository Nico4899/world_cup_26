"""Ingest the openfootball/world-cup ``cup.txt`` schedule for canonical groups.

Why this source
---------------
The Jürisoo-derived fixture list (see ``sim.fixtures``) is a list of 72 matches
without FIFA's group letters. openfootball publishes a plain-text canonical
schedule per tournament under ``github.com/openfootball/world-cup``; for WC
2026 the file is at ``2026--usa/cup.txt``. Parsing it gives us the
{group letter → tuple of 4 team names} mapping the simulator expects.

Format
------
The relevant section looks like::

    = World Cup 2026 =


    Group A:
     1. Mexico
     2. Team2
     3. Team3
     4. Team4


    Group B:
     1. ...

The parser is line-oriented and tolerant of:
    * different leading whitespace
    * optional 3-letter country codes between the number and the name
      (e.g. " 1. MEX  Mexico")
    * extra blank lines between groups

Match-day blocks that follow the group definitions are skipped — we only
extract group → team membership.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import requests
import requests_cache

from wc2026.sim.fixtures import EXPECTED_GROUPS, EXPECTED_TEAMS_PER_GROUP, GroupAssignment

DEFAULT_URL = "https://raw.githubusercontent.com/openfootball/world-cup/master/2026--usa/cup.txt"
DEFAULT_TARGET = Path("data/raw/openfootball")
DEFAULT_CACHE = Path("data/raw/openfootball/.http_cache")
DEFAULT_CACHE_EXPIRY_SECONDS = 7 * 24 * 3600

USER_AGENT = (
    "wc2026-predictor/0.1 "
    "(+https://github.com/Nico4899/world_cup_26; nico.fliegel@gmail.com) "
    "personal-research; calibrated WC 2026 predictions"
)

_HEADER_RE = re.compile(r"^\s*Group\s+([A-L])\s*:\s*$")
# " 1.  MEX  Mexico"  or  " 2. Mexico"  →  trailing run of words after any leading
# count + optional 3-letter code.
_TEAM_RE = re.compile(
    r"""^\s*
        \d+\.\s+                # leading "N." (required to distinguish from comments)
        (?:[A-Z]{2,3}\s+)?      # optional 3-letter country code, then whitespace
        (?P<name>.+?)           # team name (lazy, until end-of-line)
        \s*$""",
    re.VERBOSE,
)

logger = logging.getLogger(__name__)


def parse_cup_txt(text: str) -> dict[str, tuple[str, ...]]:
    """Return ``{group_letter: (team1, team2, team3, team4)}`` from cup.txt content.

    Raises ``ValueError`` if the parsed result is not exactly 12 groups of 4
    distinct teams.
    """
    groups: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        header_match = _HEADER_RE.match(raw_line)
        if header_match:
            current = header_match.group(1)
            groups.setdefault(current, [])
            continue
        if current is None:
            continue
        stripped = raw_line.strip()
        if not stripped:
            # Empty line — close out the current group; subsequent text must start
            # a new "Group X:" header to be captured.
            if groups[current]:
                current = None
            continue
        team_match = _TEAM_RE.match(raw_line)
        if not team_match:
            # Some other indented prose between groups — ignore but keep collecting.
            continue
        groups[current].append(team_match.group("name").strip())

    if set(groups.keys()) != {chr(ord("A") + i) for i in range(EXPECTED_GROUPS)}:
        raise ValueError(f"openfootball cup.txt: expected groups A..L; got {sorted(groups.keys())}")
    bad_size = {
        g: len(members) for g, members in groups.items() if len(members) != EXPECTED_TEAMS_PER_GROUP
    }
    if bad_size:
        raise ValueError(
            f"openfootball cup.txt: each group must have {EXPECTED_TEAMS_PER_GROUP} teams; got {bad_size}"
        )

    all_teams = [t for members in groups.values() for t in members]
    if len(set(all_teams)) != len(all_teams):
        dupes = {t for t in all_teams if all_teams.count(t) > 1}
        raise ValueError(f"openfootball cup.txt: duplicate teams across groups: {sorted(dupes)}")

    return {letter: tuple(members) for letter, members in groups.items()}


def build_group_assignment(
    text: str,
    *,
    citation: str = "openfootball/world-cup cup.txt",
) -> GroupAssignment:
    """Wrap ``parse_cup_txt`` into the dataclass the simulator consumes."""
    return GroupAssignment(groups=parse_cup_txt(text), citation=citation)


def _make_session(
    cache_path: Path | None = DEFAULT_CACHE,
    expire_seconds: int = DEFAULT_CACHE_EXPIRY_SECONDS,
) -> requests.Session:
    if cache_path is None:
        session: requests.Session = requests.Session()
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        session = requests_cache.CachedSession(
            cache_name=str(cache_path),
            backend="sqlite",
            expire_after=expire_seconds,
            allowable_codes=(200,),
            stale_if_error=True,
        )
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/plain,*/*"})
    return session


def fetch_cup_txt(
    url: str = DEFAULT_URL,
    *,
    session: requests.Session | None = None,
    target_dir: Path = DEFAULT_TARGET,
    today: datetime | None = None,
) -> Path:
    """Download cup.txt and write a dated snapshot.

    Returns the path to the new snapshot file.
    """
    today = today or datetime.now(UTC)
    target_dir.mkdir(parents=True, exist_ok=True)
    sess = session or _make_session()
    resp = sess.get(url, timeout=30)
    resp.raise_for_status()
    out = target_dir / f"cup_{today:%Y-%m-%d}.txt"
    out.write_text(resp.text, encoding="utf-8")
    return out


def load_latest_assignment(target_dir: Path = DEFAULT_TARGET) -> GroupAssignment | None:
    """Load the most recent cup_*.txt snapshot and return its GroupAssignment.

    Returns ``None`` if no snapshot exists.
    """
    paths = sorted(target_dir.glob("cup_*.txt"))
    if not paths:
        return None
    text = paths[-1].read_text(encoding="utf-8")
    return build_group_assignment(
        text, citation=f"openfootball/world-cup cup.txt (snapshot {paths[-1].stem})"
    )


def write_group_assignment_json(
    assignment: GroupAssignment,
    out_path: Path = Path("data/wc2026_group_assignment.json"),
) -> Path:
    """Serialise an assignment to the JSON form the simulator's ``load_group_assignment`` reads."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Mapping[str, object] = {
        "source": assignment.citation,
        "groups": {letter: list(members) for letter, members in assignment.groups.items()},
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


__all__ = [
    "DEFAULT_CACHE",
    "DEFAULT_CACHE_EXPIRY_SECONDS",
    "DEFAULT_TARGET",
    "DEFAULT_URL",
    "build_group_assignment",
    "fetch_cup_txt",
    "load_latest_assignment",
    "parse_cup_txt",
    "write_group_assignment_json",
]
