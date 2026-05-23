"""One-shot script: download the Jürisoo intl-results dataset and print summary.

Usage:
    uv run python scripts/download_kaggle_intl.py
    uv run python scripts/download_kaggle_intl.py --force

Requires ~/.kaggle/kaggle.json with a Kaggle API token (kaggle.com/settings → 'Create New Token').
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wc2026.ingest.kaggle_intl import (
    DEFAULT_TARGET,
    basic_stats,
    download_dataset,
    load_results,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default=str(DEFAULT_TARGET),
        help=f"Download target dir (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if results.csv already exists.",
    )
    args = parser.parse_args()

    target = Path(args.target)
    paths = download_dataset(target, force=args.force)
    df = load_results(paths.root)
    stats = basic_stats(df)
    print(json.dumps({"target": str(paths.root), **stats}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
