# Data source licences

This project's code is **MIT-licensed** (see `LICENSE`). Each upstream data
source retains its own licence; this document summarises them so it is clear
which obligations apply where.

## Match results — Jürisoo "International football results" (Kaggle)

- **Source:** [martj42/international-football-results-from-1872-to-2017](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) (`results.csv`, `shootouts.csv`)
- **Licence:** [CC0 1.0 Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/)
- **Attribution:** not legally required, but the README cites the dataset by name as good practice.
- **Consumed by:** [`src/wc2026/ingest/kaggle_intl.py`](../src/wc2026/ingest/kaggle_intl.py) and [`src/wc2026/models/shootout.py`](../src/wc2026/models/shootout.py).
- **Notes:** mirrored to `data/raw/jurisoo/`. The dataset is community-maintained and updated
  every few months; the scheduler's `kaggle_refresh` job pulls a fresh copy daily.

## Team strength — eloratings.net (World Football Elo Ratings)

- **Source:** [eloratings.net](https://www.eloratings.net) (TSV endpoint via the site's "Download CSV" mechanism).
- **Licence:** the site does not publish a formal licence. Numbers are derivative — they are computed by the maintainer (Lars Schiefler) by applying the Elo formula to publicly-known match results — so the underlying *facts* (match outcomes) are not copyrightable. Our scraper is rate-limited to **2 TSV requests per refresh** to be a polite neighbour. We attribute the source explicitly on the dashboard's About page and in the model methodology doc.
- **Attribution:** [World Football Elo Ratings — https://www.eloratings.net](https://www.eloratings.net).
- **Consumed by:** [`src/wc2026/ingest/eloratings_scraper.py`](../src/wc2026/ingest/eloratings_scraper.py) and [`src/wc2026/models/shootout.py`](../src/wc2026/models/shootout.py).
- **Notes:** snapshots are stored under `data/raw/elo/` as dated Parquet files. If you redistribute the
  snapshot files, repeat the attribution.

## WC 2026 fixtures + live results — football-data.org

- **Source:** [football-data.org REST API](https://www.football-data.org/) (`/v4/competitions/{code}/matches`).
- **Licence:** the free tier is governed by the [football-data.org Terms of Service](https://www.football-data.org/terms-of-service). Free use is permitted for personal / educational projects with attribution; commercial use requires a paid tier. The API key is bound to a single account and must not be shared.
- **Attribution:** "Data provided by football-data.org" on any consuming UI.
- **Consumed by:** [`src/wc2026/ingest/football_data_org.py`](../src/wc2026/ingest/football_data_org.py).
- **Notes:** the key is read from the `FOOTBALL_DATA_ORG_KEY` env var. If unset, the daily
  `football_data_org_refresh` scheduler job logs a warning and no-ops; the rest of the
  pipeline continues unaffected.

## FIFA tiebreaker procedure (referenced, not bundled)

The 2026 FIFA group-stage tiebreaker chain encoded in
[`src/wc2026/sim/groups.py`](../src/wc2026/sim/groups.py) is derived from FIFA's
public regulations summary at
<https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/groups-how-teams-qualify-tie-breakers>.
Factual rule statements are not subject to copyright; the source URL is cited
in the module docstring.

## Bookmaker-comparison numbers (literature-cited, not ingested)

The Track Record page cites bookmaker closing-odds log-loss figures from:

- [Wheatcroft, E. (2019). "A profitable model for predicting the over/under market in football"](https://www.sciencedirect.com/science/article/pii/S0169207020300108) — WC 2018 figures.
- [Constantinou, A. C. (2019). "Dolores: a model that predicts football match outcomes from all over the world"](https://link.springer.com/article/10.1007/s10994-018-5703-7) — methodology benchmarks.

These are quoted as references; no bookmaker data is ingested.

---

If you fork this project for a different tournament or extend it to a new
data source, please update this file before publishing. Honesty about
provenance is part of the "calibrated honesty" framing.
