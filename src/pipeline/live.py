"""
src/pipeline/live.py
---------------------
Live World Cup 2026 prediction pipeline.

Fetches latest results, updates the model, and generates predictions
for all upcoming fixtures — ready to run once per day during the tournament.

Usage:
    uv run python -m src.pipeline.live                    # predict today's matches
    uv run python -m src.pipeline.live --matchday 2026-06-16
    uv run python -m src.pipeline.live --all              # all upcoming group stage
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from src.data.wc2026 import get_fixtures, fetch_all_wc2026_results, save_live_results
from src.features.elo import EloRatingSystem
from src.models.dixon_coles import DixonColesModel
from src.pipeline.predict import predict_match
from src.pipeline.train import (
    load_trained_models,
    train,
    models_are_stale,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)

PROCESSED_MATCHES = Path("data/processed/matches.parquet")
WC_RESULTS_CACHE  = Path("data/raw/wc2026_results.csv")
OUTPUT_DIR        = Path("outputs/predictions")


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def get_models(
    force_train: bool = False,
    force_refresh: bool = False,
) -> tuple[DixonColesModel, EloRatingSystem]:
    """
    Return fitted models — loading from disk if available and fresh,
    otherwise training from scratch.

    Parameters
    ----------
    force_train : bool  — re-train even if fresh models exist on disk
    force_refresh : bool  — fetch new WC results before training
    """
    if not force_train and not models_are_stale(max_age_days=1):
        try:
            return load_trained_models()
        except FileNotFoundError:
            pass  # fall through to training

    if models_are_stale(max_age_days=1) and not force_train:
        logger.warning(
            "Saved models are more than 1 day old. Re-training now.\n"
            "  Tip: run 'uv run python -m src.pipeline.train --refresh' each morning."
        )

    logger.info("Training models (this takes ~30–60 seconds)…")
    return train(force_refresh=force_refresh)


# ---------------------------------------------------------------------------
# Matchday prediction
# ---------------------------------------------------------------------------

def predict_matchday(
    matchday_date: str,
    dc: DixonColesModel,
    elo: EloRatingSystem,
    output_dir: Path = OUTPUT_DIR,
) -> list[dict]:
    fixtures_df = get_fixtures(phase="all", as_of=matchday_date)
    fixtures_df = fixtures_df[
        fixtures_df["date"].dt.date == pd.Timestamp(matchday_date).date()
    ]

    if fixtures_df.empty:
        logger.warning("No fixtures found for %s", matchday_date)
        return []

    models = {"dixon_coles": dc, "elo": elo}

    predictions = []
    for _, fix in fixtures_df.iterrows():
        home, away = fix["home_team"], fix["away_team"]
        match_date_str = str(fix["date"].date())
        pred = predict_match(
            models=models,
            home_team=home,
            away_team=away,
            match_date=match_date_str,
            neutral_venue=True,
        )

        predictions.append(pred)
        _print_prediction(pred)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{matchday_date}_predictions.json"
    with open(out_path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)
    logger.info("Saved %d predictions → %s", len(predictions), out_path)

    return predictions


def _print_prediction(pred: dict) -> None:
    """Pretty-print a single match prediction to stdout."""
    m = pred["match"]
    o = pred["match_outcome"]
    g = pred["goals"]
    elo_d = pred["elo"]

    home, away = m["home_team"], m["away_team"]
    host_tag = " 🏠" if m.get("host_advantage") else ""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  {home:>26}{host_tag}  vs  {away}")
    print(f"  {m['date']}   (Elo: {elo_d['home_elo']:.0f} vs {elo_d['away_elo']:.0f}  diff: {elo_d['elo_diff']:+.0f})")
    print(sep)
    print(f"  Result:  {home} win {o['home_win']:.0%}  |  Draw {o['draw']:.0%}  |  {away} win {o['away_win']:.0%}")
    print(f"  Goals:   xG {g['expected_home_goals']:.2f}–{g['expected_away_goals']:.2f}  |  O2.5 {g['over_2_5']:.0%}  |  BTTS {g['btts']:.0%}")

    if pred.get("top_scorelines"):
        lines = "  ".join(f"{h}-{a}({p:.0%})" for h, a, p in pred["top_scorelines"][:3])
        print(f"  Top scorelines: {lines}")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="WC 2026 Live Prediction Engine")
    parser.add_argument("--matchday", type=str, default=None,
                        help="Predict fixtures for this date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--all", action="store_true",
                        help="Predict all remaining group stage fixtures.")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-fetch live WC 2026 results and re-train models.")
    parser.add_argument("--train", action="store_true",
                        help="Force re-train models even if saved ones exist.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    # Load or train models
    dc, elo = get_models(force_train=args.train, force_refresh=args.refresh)

    if args.all:
        today = str(date.today())
        fixtures_df = get_fixtures(phase="group", as_of=today)
        for match_date in fixtures_df["date"].dt.strftime("%Y-%m-%d").unique():
            predict_matchday(match_date, dc, elo)
    else:
        matchday = args.matchday or str(date.today())
        predict_matchday(matchday, dc, elo)


if __name__ == "__main__":
    _cli()
