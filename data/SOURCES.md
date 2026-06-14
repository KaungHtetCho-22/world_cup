# Data Sources

## Required

### International Match Results (Dixon-Coles / Elo training)
- **Source**: Kaggle — "International football results from 1872 to 2024"
- **URL**: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
- **File**: `data/raw/results.csv`
- **How to get**: Download from Kaggle (free account required)
- **Notes**: ~50k matches with tournament type, neutral venue flag. Only matches from 2020–present are used for Dixon-Coles fitting; full history is used for Elo.

---

## Bundled (no download needed)

### WC 2026 Fixtures
- **Source**: FotMob (scraped)
- **File**: `data/raw/wc2026_fixtures.csv`
- **Notes**: All group stage and knockout fixtures with team IDs, dates, and completed scores.

### Penalty Shootouts
- **Source**: Kaggle (same dataset as results.csv)
- **File**: `data/raw/shootouts.csv`
- **Notes**: Used for knockout phase tie-breaking reference.

---

## Auto-fetched (no manual download)

### WC 2026 Live Results
- **Source**: FotMob (via `wc2026_fixtures.csv` — results parsed from `score` column)
- **File**: `data/raw/wc2026_results.csv` (auto-updated)
- **How to update**: `uv run python -m src.pipeline.train --refresh`
- **Notes**: Completed match scores are read from `wc2026_fixtures.csv` and cached here. No external API key needed.

---

## Source status summary

| Source | Required? | How to get | Used by |
|---|---|---|---|
| `results.csv` | ✅ Required | Kaggle download | Dixon-Coles, Elo |
| `wc2026_fixtures.csv` | ✅ Bundled | Already present | Live fixtures + results |
| `wc2026_results.csv` | Auto | `train --refresh` | Live updates to DC + Elo |
| `shootouts.csv` | ✅ Bundled | Already present | Reference only |

---
*Last updated: 2026-06-13*
