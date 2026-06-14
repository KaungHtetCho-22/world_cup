"""
src/features/elo.py
-------------------
Build time-respecting Elo ratings for international teams.

Usage:
    from src.features.elo import EloRatingSystem
    elo = EloRatingSystem(k_base=20)
    ratings_history = elo.fit(matches_df)  # returns long-form DataFrame
    rating = elo.get_rating("France", as_of="2022-11-20")
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Default Elo starting rating for new teams
DEFAULT_ELO = 1500.0

# K-factor multipliers by tournament weight
K_MULTIPLIERS: dict[str, float] = {
    "FIFA World Cup": 2.0,
    "UEFA Euro": 1.5,
    "Copa América": 1.5,
    "African Cup of Nations": 1.3,
    "AFC Asian Cup": 1.3,
    "FIFA World Cup qualification": 1.2,
    "Friendly": 0.5,
}

# FIFA-ranking-informed Elo seeds for all WC 2026 participants.
# Derived from June 2026 FIFA ranking points mapped to Elo scale:
#   Elo_seed = 1500 + (FIFA_points - 1500) * 0.6
# Teams not listed here will use the DEFAULT_ELO (1500).
# These seeds are ONLY used as starting values before the Elo fit;
# the full match-by-match Elo replay overwrites them quickly.
WC2026_ELO_SEEDS: dict[str, float] = {
    # Top-tier (FIFA top 10)
    "Argentina":    1950,
    "France":       1920,
    "Spain":        1910,
    "England":      1890,
    "Brazil":       1880,
    "Portugal":     1870,
    "Netherlands":  1860,
    "Belgium":      1840,
    "Germany":      1840,
    # Strong contenders (FIFA 11-25)
    "Italy":        1820,
    "Uruguay":      1800,
    "Colombia":     1800,
    "USA":          1790,
    "Mexico":       1790,
    "Croatia":      1780,
    "Morocco":      1780,
    "Japan":        1780,
    "Canada":       1770,
    "Switzerland":  1770,
    "Senegal":      1770,
    "South Korea":  1760,
    "Denmark":      1760,
    "Austria":      1750,
    "Algeria":      1750,
    "Turkey":       1740,    # historical name (Turkiye in fixtures)
    "Czech Republic": 1740, # historical name (Czechia in fixtures)
    "Ecuador":      1740,
    "Sweden":       1730,
    "Norway":       1720,
    "Iran":         1720,
    "Australia":    1710,
    "Tunisia":      1710,
    "Ivory Coast":  1700,
    "Paraguay":     1700,
    "Ghana":        1690,
    "Saudi Arabia": 1680,
    "Egypt":        1680,
    "DR Congo":     1680,
    "Qatar":        1670,
    "Peru":         1670,
    # Moderate strength (FIFA 50-80)
    "Jordan":       1660,
    "New Zealand":  1650,
    "South Africa": 1640,
    "Panama":       1640,
    "Iraq":         1640,
    "Uzbekistan":   1640,
    "Scotland":     1640,
    "Bosnia-Herzegovina": 1640,  # historical name
    "Cape Verde":   1630,
    "Haiti":        1620,
    # Minnows / debutants
    "Curaçao":      1580,    # historical name (Curacao in fixtures)
}


class EloRatingSystem:
    """
    Standard Elo rating system for international football.

    Parameters
    ----------
    k_base : float
        Base K-factor. Multiplied by tournament weight for each match.
    initial_elo : float
        Starting rating for new/unknown teams.
    """

    def __init__(self, k_base: float = 30.0, initial_elo: float = DEFAULT_ELO) -> None:
        self.k_base = k_base
        self.initial_elo = initial_elo
        self.ratings: dict[str, float] = {}
        self._history: list[dict] = []

    def _get_k(self, tournament: str) -> float:
        """Return K-factor for a given tournament."""
        for key, mult in K_MULTIPLIERS.items():
            if key in tournament:
                return self.k_base * mult
        return self.k_base

    def _expected_score(self, elo_a: float, elo_b: float) -> float:
        """Elo expected score for team A against team B."""
        return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))

    def _actual_score(self, home_goals: int, away_goals: int) -> tuple[float, float]:
        """Return (home_score, away_score) as 1/0.5/0."""
        if home_goals > away_goals:
            return 1.0, 0.0
        elif home_goals < away_goals:
            return 0.0, 1.0
        return 0.5, 0.5

    def fit(self, matches: pd.DataFrame, seed_wc_teams: bool = True) -> pd.DataFrame:
        """
        Process matches chronologically, updating ratings after each match.

        Parameters
        ----------
        matches : pd.DataFrame
            Must contain: date, home_team, away_team, home_score, away_score, tournament.
            Should be sorted by date ascending.
        seed_wc_teams : bool
            If True (default), pre-seed all WC2026 teams with FIFA-ranking-
            calibrated Elo values before the match-by-match replay.  This
            prevents unknown / renamed teams (e.g. Czechia / Czech Republic)
            from incorrectly starting at 1500 and producing extreme Elo diffs.

        Returns
        -------
        pd.DataFrame
            Long-form history: team, date, elo_before, elo_after.
        """
        # Seed ratings: WC team priors first, then fall back to initial_elo.
        if seed_wc_teams:
            self.ratings = {team: seed for team, seed in WC2026_ELO_SEEDS.items()}
        else:
            self.ratings = {}
        self._history = []

        for _, row in matches.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            tournament = str(row.get("tournament", ""))
            date = row["date"]

            elo_home = self.ratings.get(home, self.initial_elo)
            elo_away = self.ratings.get(away, self.initial_elo)

            exp_home = self._expected_score(elo_home, elo_away)
            exp_away = 1.0 - exp_home

            actual_home, actual_away = self._actual_score(
                int(row["home_score"]), int(row["away_score"])
            )

            k = self._get_k(tournament)
            new_home = elo_home + k * (actual_home - exp_home)
            new_away = elo_away + k * (actual_away - exp_away)

            self._history.append(
                {"team": home, "date": date, "elo_before": elo_home, "elo_after": new_home}
            )
            self._history.append(
                {"team": away, "date": date, "elo_before": elo_away, "elo_after": new_away}
            )

            self.ratings[home] = new_home
            self.ratings[away] = new_away

        history_df = pd.DataFrame(self._history)
        logger.info("Elo fitted on %d matches, %d teams.", len(matches), len(self.ratings))
        return history_df

    def get_rating(self, team: str, as_of: Optional[str] = None) -> float:
        """
        Return Elo rating for a team, optionally as-of a specific date.

        Parameters
        ----------
        team : str
        as_of : str or None
            ISO date string (e.g. "2022-11-20"). If None, returns latest rating.
        """
        # Fallback: WC seed if available, then generic default
        fallback = WC2026_ELO_SEEDS.get(team, self.initial_elo)

        if as_of is None:
            return self.ratings.get(team, fallback)

        history_df = pd.DataFrame(self._history)
        if history_df.empty:
            return fallback

        team_hist = history_df[
            (history_df["team"] == team) & (history_df["date"] <= pd.Timestamp(as_of))
        ]
        if team_hist.empty:
            return fallback
        return float(team_hist.iloc[-1]["elo_after"])

    def save(self, path: str = "data/processed/elo_history.parquet") -> None:
        """Save Elo history to parquet."""
        pd.DataFrame(self._history).to_parquet(path, index=False)
        logger.info("Elo history saved to %s", path)

    def sanity_check(self, expected_top: list[str] | None = None) -> None:
        """Log current top-10 ratings and warn if expected top teams are low-ranked."""
        if expected_top is None:
            expected_top = ["Brazil", "Argentina", "France", "Germany", "Spain"]
        sorted_ratings = sorted(self.ratings.items(), key=lambda x: -x[1])
        logger.info("Top 10 Elo ratings:")
        for rank, (team, rating) in enumerate(sorted_ratings[:10], 1):
            logger.info("  %2d. %-25s %.1f", rank, team, rating)

        top_teams = {t for t, _ in sorted_ratings[:15]}
        missing = [t for t in expected_top if t not in top_teams]
        if missing:
            logger.warning("Expected top teams not in top 15: %s", missing)
