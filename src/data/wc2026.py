"""
src/data/wc2026.py

World Cup 2026 fixtures and results utilities.

Source:
    data/raw/wc2026_fixtures.csv

Generated from FotMob.

No ESPN dependency.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.loader import normalise_team_name

logger = logging.getLogger(__name__)

FIXTURE_FILE = Path("data/raw/wc2026_fixtures.csv")


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
def get_fixtures(
    path="data/raw/wc2026_fixtures.csv",
    phase="all",
    as_of=None,
):
    df = pd.read_csv(
        path,
        parse_dates=["date"],
    )

    df["date"] = pd.to_datetime(df["date"])

    # Normalise team names to match historical training data
    df["home_team"] = df["home_team"].map(normalise_team_name)
    df["away_team"] = df["away_team"].map(normalise_team_name)

    if as_of:
        cutoff_date = pd.Timestamp(as_of).date()

        df = df[
            df["date"].dt.date >= cutoff_date
        ]

    if phase == "group":
        df = df[df["group"].notna()]

    elif phase == "knockout":
        df = df[df["group"].isna()]

    df["neutral_venue"] = True

    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------
def fetch_all_wc2026_results(
    fixtures_path: str | Path = FIXTURE_FILE,
) -> pd.DataFrame:
    """
    Build completed results directly from
    wc2026_fixtures.csv.

    Returns
    -------
    DataFrame
        Compatible with training pipeline.
    """

    fixtures = get_fixtures(fixtures_path)

    if fixtures.empty:
        logger.warning(
            "No fixture data available."
        )
        return pd.DataFrame()

    results = fixtures[
        fixtures["finished"] == True
    ].copy()

    if results.empty:
        logger.info(
            "No completed World Cup results found."
        )
        return pd.DataFrame()

    scores = (
        results["score"]
        .fillna("")
        .astype(str)
        .str.extract(r"(\d+)\s*-\s*(\d+)")
    )

    results["home_score"] = pd.to_numeric(
        scores[0],
        errors="coerce",
    )

    results["away_score"] = pd.to_numeric(
        scores[1],
        errors="coerce",
    )

    results = results.dropna(
        subset=[
            "home_score",
            "away_score",
        ]
    )

    results["home_score"] = (
        results["home_score"]
        .astype(int)
    )

    results["away_score"] = (
        results["away_score"]
        .astype(int)
    )

    results["neutral_venue"] = True
    results["tournament"] = "FIFA World Cup"
    results["tournament_weight"] = 1.5

    results["result"] = results.apply(
        lambda r:
            "H"
            if r["home_score"] > r["away_score"]
            else (
                "A"
                if r["home_score"] < r["away_score"]
                else "D"
            ),
        axis=1,
    )

    results["total_goals"] = (
        results["home_score"]
        + results["away_score"]
    )

    return results[
        [
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "result",
            "total_goals",
            "neutral_venue",
            "tournament",
            "tournament_weight",
        ]
    ].sort_values("date")


# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------
def save_live_results(
    df: pd.DataFrame,
    path: str = "data/raw/wc2026_results.csv",
) -> None:

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
    )

    logger.info(
        "Saved %d results to %s",
        len(df),
        path,
    )


# ---------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    fixtures = get_fixtures()

    print("\n=== FIXTURES ===")
    print(f"Count: {len(fixtures)}")

    if not fixtures.empty:
        print(
            fixtures.head()[
                [
                    "date",
                    "home_team",
                    "away_team",
                    "group",
                    "finished",
                ]
            ]
        )

    results = fetch_all_wc2026_results()

    print("\n=== RESULTS ===")
    print(f"Count: {len(results)}")

    if not results.empty:
        print(results.head())

        save_live_results(results)