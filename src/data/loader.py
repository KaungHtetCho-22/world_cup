"""
src/data/loader.py
------------------
Load and clean historical international match data.

Raw data expected at data/raw/results.csv (Kaggle international football results).
Output saved to data/processed/matches.parquet.

Column schema of the cleaned DataFrame:
    date          : datetime64[ns]
    home_team     : str
    away_team     : str
    home_score    : int
    away_score    : int
    tournament    : str
    neutral_venue : bool
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Team-name normalisation map
# Add entries as inconsistencies are discovered across data sources.
# ---------------------------------------------------------------------------
TEAM_NAME_MAP: dict[str, str] = {
    "United States": "USA",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Cape Verde Islands": "Cape Verde",
    "Trinidad and Tobago": "Trinidad & Tobago",
    # WC 2026 fixture name → historical data name
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Czechia": "Czech Republic",      # FIFA switched name in 2016; historical data uses 'Czech Republic'
    "Turkiye": "Turkey",              # FIFA rebranded; historical data uses 'Turkey'
    "Curacao": "Curaçao",             # historical data uses the accented form
}

# Tournament weight — used later in Elo / Dixon-Coles weighting
TOURNAMENT_WEIGHTS: dict[str, float] = {
    "FIFA World Cup": 1.5,
    "UEFA Euro": 1.3,
    "Copa América": 1.3,
    "African Cup of Nations": 1.2,
    "AFC Asian Cup": 1.2,
    "FIFA World Cup qualification": 1.1,
    "Friendly": 0.5,
}


def normalise_team_name(name: str) -> str:
    """Apply the canonical team-name mapping."""
    return TEAM_NAME_MAP.get(name, name)


def tournament_weight(tournament: str) -> float:
    """Return competitive weight for a given tournament name."""
    for key, weight in TOURNAMENT_WEIGHTS.items():
        if key in tournament:
            return weight
    return 1.0


def load_raw_results(raw_path: Path | str = "data/raw/results.csv") -> pd.DataFrame:
    """
    Read the raw Kaggle international results CSV.

    Returns a DataFrame with at minimum:
        date, home_team, away_team, home_score, away_score, tournament, neutral_venue
    """
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw results file not found: {raw_path}\n"
            "Download from https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017"
        )

    df = pd.read_csv(raw_path, parse_dates=["date"])

    # Rename columns to standard schema
    rename_map = {
        "home_team": "home_team",
        "away_team": "away_team",
        "home_score": "home_score",
        "away_score": "away_score",
        "tournament": "tournament",
        "neutral": "neutral_venue",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    return df


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardise the raw results DataFrame.

    Steps:
    1. Normalise team names.
    2. Drop rows with missing scores.
    3. Cast scores to int.
    4. Remove obvious duplicates.
    5. Add derived columns: result (H/D/A), total_goals, tournament_weight.
    """
    # 1. Normalise team names
    df["home_team"] = df["home_team"].map(normalise_team_name)
    df["away_team"] = df["away_team"].map(normalise_team_name)

    # 2. Drop rows with missing scores
    before = len(df)
    df = df.dropna(subset=["home_score", "away_score"])
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with missing scores.", dropped)

    # 3. Cast to int
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # 4. Remove duplicates
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"])

    # 5. Derived columns
    df["result"] = df.apply(
        lambda r: "H" if r["home_score"] > r["away_score"]
        else ("A" if r["home_score"] < r["away_score"] else "D"),
        axis=1,
    )
    df["total_goals"] = df["home_score"] + df["away_score"]
    df["tournament_weight"] = df["tournament"].apply(tournament_weight)

    df = df.sort_values("date").reset_index(drop=True)
    return df


def save_processed(df: pd.DataFrame, out_path: Path | str = "data/processed/matches.parquet") -> None:
    """Save cleaned DataFrame to parquet."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Saved %d rows to %s", len(df), out_path)


def load_processed(path: Path | str = "data/processed/matches.parquet") -> pd.DataFrame:
    """Load the cleaned processed matches parquet."""
    return pd.read_parquet(Path(path))


def build_matches(
    raw_path: str = "data/raw/results.csv",
    out_path: str = "data/processed/matches.parquet",
) -> pd.DataFrame:
    """Full pipeline: load raw → clean → save → return."""
    df = load_raw_results(raw_path)
    df = clean_matches(df)
    save_processed(df, out_path)
    logger.info("build_matches complete: %d matches.", len(df))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_matches()
