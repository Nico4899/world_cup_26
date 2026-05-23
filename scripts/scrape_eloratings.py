"""One-shot script: fetch World Football Elo ratings and write a Parquet snapshot.

Usage:
    uv run python scripts/scrape_eloratings.py
    uv run python scripts/scrape_eloratings.py --no-cache

Polite scrape: exactly two GETs per run (World.tsv + en.teams.tsv), each with a
User-Agent that names the project, links the repo, and gives a contact email.
Per-request body is ~30 KB / ~7 KB respectively.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wc2026.ingest.eloratings_scraper import (
    DEFAULT_CACHE,
    DEFAULT_TARGET,
    fetch_current_ratings,
    load_latest_snapshot,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default=str(DEFAULT_TARGET),
        help=f"Snapshot dir (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable requests-cache; always hit the network.",
    )
    args = parser.parse_args()

    target = Path(args.target)
    cache = None if args.no_cache else DEFAULT_CACHE
    out = fetch_current_ratings(target_dir=target, cache_path=cache)
    df = load_latest_snapshot(target)

    top5 = df.nsmallest(5, "global_rank")[["global_rank", "code", "team_name", "rating"]]
    print(
        json.dumps(
            {
                "snapshot_path": str(out),
                "n_teams": len(df),
                "top5": top5.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
