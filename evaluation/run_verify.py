#!/usr/bin/env python3
"""
Verification entry point for the Bitcoin evaluation experiment.

Usage:
    python -m evaluation.run_verify

This script:
1. Loads the predictions CSV.
2. Finds unverified predictions where ≥24h have elapsed.
3. For each, fetches the actual BTC price at timestamp + 24h.
4. Computes error metrics and updates the CSV row.
5. Regenerates the Markdown report.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta, timezone

from .csv_manager import load_predictions, update_verification
from .price_verifier import get_verified_price
from .metrics import (
    compute_absolute_error,
    compute_direction,
    compute_percentage_error,
    is_direction_correct,
    is_within_range,
)
from .report import generate_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    # 1. Load predictions
    df = load_predictions()
    logger.info("Loaded %d prediction(s) from CSV", len(df))

    if len(df) == 0:
        logger.info("No predictions to verify. Generating empty report.")
        generate_report(df)
        return 0

    # 2. Find unverified rows where ≥24h have elapsed
    now_utc = datetime.now(timezone.utc)
    verified_count = 0
    error_count = 0

    for idx, row in df.iterrows():
        # Skip already-verified rows
        actual = row.get("actual_price_24h", "")
        if actual and actual != "" and actual != "nan":
            continue

        # Parse prediction timestamp
        ts_str = row.get("timestamp_utc", "")
        if not ts_str:
            continue

        try:
            pred_time = datetime.fromisoformat(
                ts_str.replace("Z", "+00:00")
            )
        except ValueError:
            logger.warning("Cannot parse timestamp: %s", ts_str)
            continue

        # Check if ≥24h have elapsed
        verify_time = pred_time + timedelta(hours=24)
        if now_utc < verify_time:
            hours_left = (verify_time - now_utc).total_seconds() / 3600
            logger.info(
                "Prediction %s: %.1fh remaining until verification",
                ts_str,
                hours_left,
            )
            continue

        # 3. Fetch actual price at prediction_time + 24h (multi-source)
        logger.info(
            "Verifying prediction from %s (target: %s)...",
            ts_str,
            verify_time.isoformat(),
        )
        try:
            verified = get_verified_price(verify_time)
            actual_price = verified.price
        except Exception as exc:
            logger.error(
                "Failed to fetch actual price for %s: %s", ts_str, exc
            )
            error_count += 1
            continue

        # 4. Compute metrics
        initial_price = float(row["initial_price"])
        predicted_price = float(row["predicted_price_24h"])
        predicted_min = float(row["predicted_min"])
        predicted_max = float(row["predicted_max"])
        predicted_dir = row.get("predicted_direction", "")

        abs_error = compute_absolute_error(predicted_price, actual_price)
        pct_error = compute_percentage_error(predicted_price, actual_price)
        actual_dir = compute_direction(initial_price, actual_price)
        dir_correct = is_direction_correct(predicted_dir, actual_dir)
        in_range = is_within_range(actual_price, predicted_min, predicted_max)

        # 5. Update CSV row (with multi-source data)
        updated = update_verification(
            timestamp_utc=ts_str,
            actual_price=actual_price,
            absolute_error=abs_error,
            percentage_error=pct_error,
            actual_direction=actual_dir,
            direction_correct=dir_correct,
            within_range=in_range,
            source_coingecko=verified.source_coingecko,
            source_binance=verified.source_binance,
            price_confidence=verified.confidence,
        )
        if updated:
            verified_count += 1
            logger.info(
                "  Actual: $%,.2f | Error: $%,.2f (%.2f%%) | "
                "Dir: %s→%s (%s) | Range: %s | Confidence: %s",
                actual_price,
                abs_error,
                pct_error,
                predicted_dir,
                actual_dir,
                "✓" if dir_correct else "✗",
                "✓" if in_range else "✗",
                verified.confidence,
            )

    # 6. Regenerate report with updated data
    logger.info(
        "Verification complete: %d verified, %d errors",
        verified_count,
        error_count,
    )
    updated_df = load_predictions()
    generate_report(updated_df)

    return 0


if __name__ == "__main__":
    sys.exit(main())
