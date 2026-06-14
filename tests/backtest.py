"""
tests/backtest.py
-----------------
Backtesting harness for the Dixon-Coles model.

Evaluates predictions against actual outcomes from a past World Cup
(2018 or 2022) using:
  - Log-loss
  - Brier score
  - Calibration check (predicted probability bucket vs. actual frequency)

Results are saved to tests/backtest_results.md.

Usage:
    uv run python tests/backtest.py --wc 2022 --matches data/processed/matches.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from src.data.loader import load_processed
from src.features.elo import EloRatingSystem
from src.models.dixon_coles import DixonColesModel

logger = logging.getLogger(__name__)

WC_YEARS = {
    2018: ("2018-06-14", "2018-07-15"),
    2022: ("2022-11-20", "2022-12-18"),
}

RESULTS_PATH = Path("tests/backtest_results.md")


def backtest(matches_df: pd.DataFrame, wc_year: int) -> dict:
    """
    Run backtest for a given World Cup year.

    Training data: everything BEFORE the tournament start.
    Test data: World Cup matches from that year.

    Returns
    -------
    dict with log_loss, brier_home, brier_draw, brier_away, n_matches, calibration
    """
    if wc_year not in WC_YEARS:
        raise ValueError(f"Unknown WC year: {wc_year}. Options: {list(WC_YEARS)}")

    start, end = WC_YEARS[wc_year]
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    train = matches_df[matches_df["date"] < start_ts]
    test = matches_df[
        (matches_df["date"] >= start_ts) & (matches_df["date"] <= end_ts)
        & (matches_df["tournament"].str.contains("FIFA World Cup", na=False))
        & (~matches_df["tournament"].str.contains("qualification", case=False, na=False))
    ]

    if test.empty:
        logger.warning("No WC %d matches found in dataset.", wc_year)
        return {}

    logger.info("Training on %d matches before %s, testing on %d WC %d matches.",
                len(train), start, len(test), wc_year)

    # Fit model on training data only
    dc = DixonColesModel(half_life_days=365)
    dc.fit(train, reference_date=start)

    # Generate predictions for each test match
    y_true_1x2 = []     # one-hot encoded [home, draw, away]
    y_pred_1x2 = []
    home_probs, draw_probs, away_probs = [], [], []
    actual_home, actual_draw, actual_away = [], [], []

    for _, row in test.iterrows():
        pred = dc.predict(row["home_team"], row["away_team"])

        p_home = pred["home_win"]
        p_draw = pred["draw"]
        p_away = pred["away_win"]

        # Actual result
        if row["home_score"] > row["away_score"]:
            actual = [1, 0, 0]
        elif row["home_score"] == row["away_score"]:
            actual = [0, 1, 0]
        else:
            actual = [0, 0, 1]

        y_true_1x2.append(actual)
        y_pred_1x2.append([p_home, p_draw, p_away])
        home_probs.append(p_home)
        draw_probs.append(p_draw)
        away_probs.append(p_away)
        actual_home.append(actual[0])
        actual_draw.append(actual[1])
        actual_away.append(actual[2])

    y_true_1x2 = np.array(y_true_1x2)
    y_pred_1x2 = np.array(y_pred_1x2)

    ll = log_loss(y_true_1x2, y_pred_1x2, labels=[0, 1, 2])
    brier_home = brier_score_loss(actual_home, home_probs)
    brier_draw = brier_score_loss(actual_draw, draw_probs)
    brier_away = brier_score_loss(actual_away, away_probs)

    # Calibration: bucket predictions into deciles
    calibration = _calibration_check(np.array(home_probs), np.array(actual_home))

    results = {
        "wc_year": wc_year,
        "n_matches": len(test),
        "log_loss": round(ll, 4),
        "brier_home": round(brier_home, 4),
        "brier_draw": round(brier_draw, 4),
        "brier_away": round(brier_away, 4),
        "calibration": calibration,
    }

    logger.info(
        "WC %d — log_loss=%.4f, brier_home=%.4f, brier_draw=%.4f, brier_away=%.4f",
        wc_year, ll, brier_home, brier_draw, brier_away,
    )

    return results


def _calibration_check(probs: np.ndarray, actuals: np.ndarray, n_bins: int = 5) -> list[dict]:
    """Bucket predictions and check actual vs. predicted frequencies."""
    bins = np.linspace(0, 1, n_bins + 1)
    calibration = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        calibration.append({
            "bucket": f"{lo:.1f}–{hi:.1f}",
            "n": int(mask.sum()),
            "pred_mean": round(float(probs[mask].mean()), 3),
            "actual_freq": round(float(actuals[mask].mean()), 3),
        })
    return calibration


def write_results(results: dict, path: Path = RESULTS_PATH) -> None:
    """Append backtest results to markdown file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a") as f:
        f.write(f"\n## Backtest: WC {results['wc_year']} — Dixon-Coles Model\n\n")
        f.write(f"- **Matches**: {results['n_matches']}\n")
        f.write(f"- **Log-loss**: {results['log_loss']}\n")
        f.write(f"- **Brier (home win)**: {results['brier_home']}\n")
        f.write(f"- **Brier (draw)**: {results['brier_draw']}\n")
        f.write(f"- **Brier (away win)**: {results['brier_away']}\n\n")
        f.write("### Calibration (home win probability)\n\n")
        f.write("| Bucket | N | Predicted | Actual |\n")
        f.write("|--------|---|-----------|--------|\n")
        for row in results.get("calibration", []):
            f.write(
                f"| {row['bucket']} | {row['n']} | {row['pred_mean']} | {row['actual_freq']} |\n"
            )
        f.write("\n---\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Backtest Dixon-Coles model against past World Cups")
    parser.add_argument("--wc", type=int, choices=[2018, 2022], default=2022)
    parser.add_argument("--matches", type=str, default="data/processed/matches.parquet")
    args = parser.parse_args()

    matches_df = load_processed(args.matches)
    results = backtest(matches_df, args.wc)
    if results:
        write_results(results)
        logger.info("Backtest results written to %s", RESULTS_PATH)
