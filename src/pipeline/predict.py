"""
src/pipeline/predict.py
------------------------
Core single-match prediction helper for WC 2026.

Called by live.py for each fixture. Not a standalone entry point.

Usage:
    from src.pipeline.predict import predict_match
    pred = predict_match(models, home_team="France", away_team="Argentina",
                         match_date="2026-06-26", neutral_venue=True)
"""

from __future__ import annotations

import logging
from datetime import date

from src.features.elo import EloRatingSystem
from src.models.dixon_coles import DixonColesModel

logger = logging.getLogger(__name__)

# WC 2026 host nations — they receive a genuine home-advantage boost.
HOST_NATIONS = {"USA", "Canada", "Mexico"}


def _is_host_match(home_team: str, away_team: str) -> bool:
    """Return True if one of the host nations is playing at home."""
    return home_team in HOST_NATIONS


def _elo_win_probabilities(home_elo: float, away_elo: float) -> tuple[float, float, float]:
    """
    Convert Elo ratings to (home_win, draw, away_win) probabilities.

    Uses the standard Elo expected-score formula and splits the remainder
    equally into draw probability following the Bradley-Terry approach
    adopted by FiveThirtyEight and most football rating systems.
    """
    elo_diff = home_elo - away_elo
    # Elo win probability for home side
    elo_home_win = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
    elo_away_win = 1.0 / (1.0 + 10 ** (elo_diff / 400.0))

    # Draw correction: international football has ~25% draw rate at neutral venue.
    # Scale down win probs proportionally to make room for draws.
    # Empirically calibrated: draw_p ~ 0.28 * exp(-|elo_diff| / 600)
    import math
    draw_p = 0.28 * math.exp(-abs(elo_diff) / 600.0)
    scale = 1.0 - draw_p
    home_win = elo_home_win * scale
    away_win = elo_away_win * scale
    return home_win, draw_p, away_win


def predict_match(
    models: dict,
    home_team: str,
    away_team: str,
    match_date: str | None = None,
    neutral_venue: bool = True,
) -> dict:
    """
    Generate full prediction for a single WC 2026 match.

    Parameters
    ----------
    models : dict  — must contain keys: 'dixon_coles', 'elo'
    home_team, away_team : str
    match_date : ISO date string or None
    neutral_venue : bool
        Default True for all WC matches, but overridden to False
        when a host nation (USA / Canada / Mexico) is the home team.

    Returns
    -------
    dict with keys: match, elo, match_outcome, goals, top_scorelines
    """
    dc: DixonColesModel = models["dixon_coles"]
    elo: EloRatingSystem = models["elo"]

    # Host nations are NOT at a neutral venue — they get home advantage.
    effective_neutral = neutral_venue and not _is_host_match(home_team, away_team)

    if neutral_venue and not effective_neutral:
        logger.info(
            "Host nation detected (%s) — applying home advantage (neutral_venue=False).",
            home_team,
        )

    # Core match prediction from Dixon-Coles
    core = dc.predict(home_team, away_team, neutral_venue=effective_neutral)

    # Elo ratings
    home_elo = elo.get_rating(home_team, as_of=match_date)
    away_elo = elo.get_rating(away_team, as_of=match_date)

    # -----------------------------------------------------------------------
    # Elo-blended outcome probabilities
    #
    # Dixon-Coles is the primary model (captures scoring rates, rho correction),
    # but when teams have sparse match data, its probabilities can be extreme.
    # Blending 30% Elo signal keeps predictions calibrated.
    #
    # Adaptive blend: if DC is very confident (> 80%) AND Elo disagrees by
    # more than 15 percentage points, we trust Elo more (up to 50% weight).
    # This specifically handles the "team X has no historical data" case.
    # -----------------------------------------------------------------------
    dc_home = core["home_win"]
    dc_draw = core["draw"]
    dc_away = core["away_win"]

    elo_home, elo_draw, elo_away = _elo_win_probabilities(home_elo, away_elo)

    # Base blend weights
    dc_weight = 0.70
    elo_weight = 0.30

    # Adaptive: if DC is very one-sided and Elo disagrees strongly, trust Elo more
    dc_extreme = max(dc_home, dc_away) > 0.80
    elo_disagrees = abs(dc_home - elo_home) > 0.15
    if dc_extreme and elo_disagrees:
        dc_weight = 0.50
        elo_weight = 0.50
        logger.info(
            "Adaptive Elo blend activated for %s vs %s "
            "(DC=%.0f%% vs Elo=%.0f%% home win). Using 50/50 blend.",
            home_team, away_team, dc_home * 100, elo_home * 100,
        )

    blended_home = dc_weight * dc_home + elo_weight * elo_home
    blended_draw = dc_weight * dc_draw + elo_weight * elo_draw
    blended_away = dc_weight * dc_away + elo_weight * elo_away

    # Re-normalise to ensure sum = 1.0
    total = blended_home + blended_draw + blended_away
    blended_home /= total
    blended_draw /= total
    blended_away /= total

    return {
        "match": {
            "home_team": home_team,
            "away_team": away_team,
            "date": match_date or str(date.today()),
            "neutral_venue": effective_neutral,
            "host_advantage": not effective_neutral,
        },
        "elo": {
            "home_elo": round(home_elo, 1),
            "away_elo": round(away_elo, 1),
            "elo_diff": round(home_elo - away_elo, 1),
        },
        "match_outcome": {
            "home_win": round(blended_home, 4),
            "draw":     round(blended_draw, 4),
            "away_win": round(blended_away, 4),
        },
        "goals": {
            "expected_home_goals": round(core["expected_home_goals"], 3),
            "expected_away_goals": round(core["expected_away_goals"], 3),
            "over_2_5":  round(core["over_2_5"], 4),
            "under_2_5": round(core["under_2_5"], 4),
            "btts":      round(core["btts"], 4),
        },
        "top_scorelines": dc.top_scorelines(core, n=5),
    }
