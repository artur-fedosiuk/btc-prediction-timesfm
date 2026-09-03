#!/usr/bin/env python3
"""
Prediction entry point for the Bitcoin evaluation experiment.

Usage:
    # With ARIMA fallback (for CI / GitHub Actions)
    python -m evaluation.run_predict

    # With real TimesFM 3.0 (local, requires torch + timesfm3)
    python -m evaluation.run_predict --use-timesfm

This script:
1. Checks if the experiment is still within the configured window.
2. Fetches the current BTC/USD price from CoinGecko.
3. Fetches 90 days of hourly price history.
4. Runs the prediction (TimesFM or ARIMA fallback).
5. Saves the prediction row to the CSV file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

import numpy as np

from .coingecko import get_btc_history, get_current_btc_price
from .config import is_experiment_active
from .csv_manager import save_prediction
from .metrics import compute_direction
from .predictor import predict_with_fallback, predict_with_timesfm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run BTC/USD 24h prediction"
    )
    parser.add_argument(
        "--use-timesfm",
        action="store_true",
        help="Use real TimesFM 3.0 model (requires torch + timesfm3)",
    )
    args = parser.parse_args()

    # 1. Check experiment window
    if not is_experiment_active():
        logger.info(
            "Experiment window has ended or not started. Exiting cleanly."
        )
        return 0

    # 2. Fetch current price
    logger.info("Fetching current BTC/USD price from CoinGecko...")
    current_price = get_current_btc_price()
    logger.info("Current BTC/USD price: $%,.2f", current_price)

    # 3. Fetch price history
    logger.info("Fetching 90 days of BTC/USD hourly history...")
    history_tuples = get_btc_history(days=90)
    history_prices = np.array(
        [price for _, price in history_tuples], dtype=np.float64
    )
    logger.info("Retrieved %d hourly data points", len(history_prices))

    if len(history_prices) < 30:
        logger.error(
            "Insufficient historical data (%d points, need ≥30). Exiting.",
            len(history_prices),
        )
        return 1

    # 4. Run prediction
    if args.use_timesfm:
        logger.info("Running TimesFM 3.0 prediction...")
        result = predict_with_timesfm(history_prices)
    else:
        logger.info("Running ARIMA fallback prediction...")
        result = predict_with_fallback(history_prices)

    # 5. Compute predicted direction
    predicted_direction = compute_direction(current_price, result.forecast)

    # 6. Save to CSV
    now_utc = datetime.now(timezone.utc)
    row = {
        "timestamp_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "initial_price": f"{current_price:.2f}",
        "predicted_price_24h": f"{result.forecast:.2f}",
        "predicted_min": f"{result.min_price:.2f}",
        "predicted_max": f"{result.max_price:.2f}",
        "actual_price_24h": "",
        "absolute_error": "",
        "percentage_error": "",
        "predicted_direction": predicted_direction,
        "actual_direction": "",
        "direction_correct": "",
        "within_range": "",
        "prediction_method": result.method,
    }
    save_prediction(row)

    logger.info(
        "Prediction saved: $%,.2f → $%,.2f (%s, method=%s)",
        current_price,
        result.forecast,
        predicted_direction,
        result.method,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
