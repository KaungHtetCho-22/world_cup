"""
src/pipeline/train.py
----------------------
Explicit model training step for the WC 2026 prediction engine.

Fits Dixon-Coles and Elo on historical + live data, then serializes
the trained models to outputs/models/ so predict/live.py can load them
instantly without re-fitting every run.

Default training parameters (calibrated against WC 2018 / 2022 backtests):
  - cutoff_years=8       ~7,300 matches; covers full 2018+2022 WC cycles
  - half_life_days=270   9-month half-life; recent form weighted 2× older data
  - min_tournament_weight=0.6  friendlies included at low weight — important
                               signal for smaller national teams with few games
  - shrinkage_prior=15   stronger regularisation pulls sparse teams to mean
  - Elo k_base=32        slightly more responsive than default 30

Note: shorter cutoff (e.g. 6 years) or more aggressive half-life (180 days)
causes many teams to have near-zero training samples, collapsing them to
default Elo 1500 and producing wildly overconfident/underconfident odds.

Usage:
    uv run python -m src.pipeline.train             # fit on all data up to today
    uv run python -m src.pipeline.train --refresh   # fetch latest WC results first
    uv run python -m src.pipeline.train --cutoff 8  # explicit cutoff
"""

from __future__ import annotations

import argparse
import logging
import pickle
import time
from datetime import date
from pathlib import Path

import pandas as pd

from src.data.wc2026 import fetch_all_wc2026_results, save_live_results
from src.data.loader import load_processed
from src.features.elo import EloRatingSystem
from src.models.dixon_coles import DixonColesModel

logger = logging.getLogger(__name__)

PROCESSED_MATCHES = Path("data/processed/matches.parquet")
WC_RESULTS_CACHE  = Path("data/raw/wc2026_results.csv")
MODELS_DIR        = Path("outputs/models")
DC_MODEL_PATH     = MODELS_DIR / "dixon_coles.pkl"
ELO_MODEL_PATH    = MODELS_DIR / "elo.pkl"
TRAIN_META_PATH   = MODELS_DIR / "train_meta.txt"


def load_training_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Build the full training dataset:
    historical matches + any completed WC 2026 matches so far.
    """
    if not PROCESSED_MATCHES.exists():
        raise FileNotFoundError(
            f"Processed matches not found at {PROCESSED_MATCHES}.\n"
            "Run first:  uv run python -m src.data.loader"
        )

    historical = load_processed(PROCESSED_MATCHES)
    logger.info("Historical base: %d matches", len(historical))

    # Fetch / reload WC 2026 results
    if force_refresh or not WC_RESULTS_CACHE.exists():
        logger.info("Fetching live WC 2026 results…")
        live_df = fetch_all_wc2026_results()
        if not live_df.empty:
            save_live_results(live_df)
            logger.info("  %d WC 2026 results fetched and cached.", len(live_df))
        else:
            logger.info("  No completed WC 2026 results yet.")
    else:
        live_df = pd.read_csv(WC_RESULTS_CACHE, parse_dates=["date"])
        logger.info("WC 2026 results from cache: %d matches", len(live_df))

    if not live_df.empty:
        # Remove any WC 2026 rows that might already be in the historical set
        wc2026_mask = (
            historical["tournament"].str.contains("FIFA World Cup", na=False)
            & (historical["date"] >= pd.Timestamp("2026-06-11"))
        )
        historical = historical[~wc2026_mask]

        # WC 2026 results get the highest tournament weight so the model
        # learns from them immediately during the group stage.
        live_df["tournament_weight"] = 2.0

        # Normalise timezones: historical parquet is tz-naive, live CSV may be tz-aware.
        if hasattr(live_df["date"].dtype, "tz") and live_df["date"].dt.tz is not None:
            live_df["date"] = live_df["date"].dt.tz_localize(None)
        if hasattr(historical["date"].dtype, "tz") and historical["date"].dt.tz is not None:
            historical["date"] = historical["date"].dt.tz_localize(None)

        combined = pd.concat([historical, live_df], ignore_index=True).sort_values("date")
        logger.info("Training dataset: %d matches total (%d WC 2026)", len(combined), len(live_df))
        return combined

    return historical


def train(
    force_refresh: bool = False,
    cutoff_years: int = 8,
    min_tournament_weight: float = 0.6,
) -> tuple[DixonColesModel, EloRatingSystem]:
    """
    Fit Dixon-Coles and Elo models and save them to disk.

    Parameters
    ----------
    force_refresh : bool
        Re-fetch live WC 2026 results before training.
    cutoff_years : int
        Years of history to use for Dixon-Coles fitting (default 8).
        8 years → ~7,300 matches, covering WC 2018 + 2022 cycles.
        Do not go below 7 — smaller windows leave many teams with too
        few matches, degrading accuracy significantly.
    min_tournament_weight : float
        Minimum tournament weight to include (default 0.6).
        0.6 keeps friendlies (weight=0.5) which are important for
        smaller nations with few competitive fixtures.

    Returns
    -------
    (DixonColesModel, EloRatingSystem)  — both are also saved to outputs/models/
    """
    matches = load_training_data(force_refresh=force_refresh)

    ref_date = str(date.today())

    # --- Dixon-Coles ---
    logger.info(
        "Fitting Dixon-Coles (cutoff=%d years, min_weight=%.1f, half_life=270d, shrinkage=15)…",
        cutoff_years, min_tournament_weight,
    )
    t0 = time.perf_counter()
    dc = DixonColesModel(half_life_days=270, shrinkage_prior=15)  # 9-month half-life; stronger shrinkage
    dc.fit(
        matches,
        reference_date=ref_date,
        cutoff_years=cutoff_years,
        min_tournament_weight=min_tournament_weight,
    )
    dc_time = time.perf_counter() - t0
    logger.info(
        "Dixon-Coles fitted in %.1fs — %d teams, home_adv=%.3f, rho=%.3f",
        dc_time, len(dc.teams_), dc.params_["home_advantage"], dc.params_["rho"],
    )

    # --- Elo (uses ALL history for accurate long-run ratings) ---
    logger.info("Fitting Elo ratings on full history…")
    t1 = time.perf_counter()
    elo = EloRatingSystem(k_base=32)   # slightly more responsive than default 30
    elo.fit(matches)
    elo.sanity_check()
    logger.info("Elo fitted in %.1fs — %d teams", time.perf_counter() - t1, len(elo.ratings))

    # --- Save to disk ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(DC_MODEL_PATH, "wb") as f:
        pickle.dump(dc, f)
    with open(ELO_MODEL_PATH, "wb") as f:
        pickle.dump(elo, f)

    meta = (
        f"trained_at: {date.today()}\n"
        f"n_matches_total: {len(matches)}\n"
        f"cutoff_years: {cutoff_years}\n"
        f"min_tournament_weight: {min_tournament_weight}\n"
        f"half_life_days: 270\n"
        f"shrinkage_prior: 15\n"
        f"dc_fit_seconds: {dc_time:.1f}\n"
        f"n_teams_dc: {len(dc.teams_)}\n"
        f"home_advantage: {dc.params_['home_advantage']:.4f}\n"
        f"rho: {dc.params_['rho']:.4f}\n"
    )
    TRAIN_META_PATH.write_text(meta)

    logger.info("Models saved → %s", MODELS_DIR)
    logger.info("Training metadata:\n%s", meta)

    return dc, elo


def load_trained_models() -> tuple[DixonColesModel, EloRatingSystem]:
    """
    Load pre-trained models from disk.

    Raises FileNotFoundError if models haven't been trained yet.
    """
    if not DC_MODEL_PATH.exists() or not ELO_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained models not found. Run the training step first:\n"
            "  uv run python -m src.pipeline.train"
        )

    with open(DC_MODEL_PATH, "rb") as f:
        dc: DixonColesModel = pickle.load(f)
    with open(ELO_MODEL_PATH, "rb") as f:
        elo: EloRatingSystem = pickle.load(f)

    meta = TRAIN_META_PATH.read_text() if TRAIN_META_PATH.exists() else "(no metadata)"
    logger.info("Loaded pre-trained models.\n%s", meta)

    return dc, elo


def models_are_stale(max_age_days: int = 1) -> bool:
    """Return True if models were trained more than max_age_days ago."""
    if not TRAIN_META_PATH.exists():
        return True
    meta = TRAIN_META_PATH.read_text()
    for line in meta.splitlines():
        if line.startswith("trained_at:"):
            trained = pd.Timestamp(line.split(":", 1)[1].strip())
            age = (pd.Timestamp(date.today()) - trained).days
            return age >= max_age_days
    return True


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train WC 2026 prediction models")
    parser.add_argument("--refresh", action="store_true",
                        help="Fetch latest WC 2026 results before training.")
    parser.add_argument("--cutoff", type=int, default=8,
                        help="Years of history to use for Dixon-Coles fitting (default: 8).")
    parser.add_argument("--min-weight", type=float, default=0.6,
                        help="Minimum tournament weight to include (default: 0.6, keeps friendlies).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    train(
        force_refresh=args.refresh,
        cutoff_years=args.cutoff,
        min_tournament_weight=args.min_weight,
    )
    print("\nDone. Run predictions with:")
    print("  uv run python -m src.pipeline.live")


if __name__ == "__main__":
    _cli()
