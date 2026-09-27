"""
CSV manager for Bitcoin prediction data.

Handles loading, appending predictions, and updating verification results.
Never overwrites or deletes existing rows.
"""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path

import pandas as pd

from .config import CSV_COLUMNS, CSV_PATH
from .forward.storage import atomic_write

logger = logging.getLogger(__name__)


def _ensure_csv(path: Path | None = None) -> Path:
    """Ensure the CSV file and its parent directory exist with headers."""
    path = path or CSV_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
        logger.info("Created CSV with headers: %s", path)
    return path


def load_predictions(path: Path | None = None) -> pd.DataFrame:
    """Load the predictions CSV into a DataFrame.

    Creates the file with headers if it doesn't exist.

    Args:
        path: Optional override for the CSV path.

    Returns:
        DataFrame with all prediction rows.
    """
    path = _ensure_csv(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # Ensure all expected columns exist
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # Apply defaults for legacy rows (before 1.5)
    if "experiment_version" in df.columns:
        df.loc[df["experiment_version"] == "", "experiment_version"] = "legacy_h1"
    if "forecast_horizon" in df.columns:
        df.loc[df["forecast_horizon"] == "", "forecast_horizon"] = "1"

    return df


def _assert_writable_schema(path: Path) -> None:
    if path.resolve() == CSV_PATH.resolve():
        raise ValueError("Legacy archive is read-only; use evaluation.forward.storage")
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    if (
        not rows
        or rows[0] != CSV_COLUMNS
        or any(len(row) != len(CSV_COLUMNS) for row in rows[1:])
    ):
        raise ValueError(
            "CSV schema mismatch; refusing to modify legacy or corrupt data"
        )


def save_prediction(row: dict, path: Path | None = None) -> None:
    """Append a single prediction row to the CSV.

    Args:
        row: Dictionary with keys matching CSV_COLUMNS.
        path: Optional override for the CSV path.
    """
    path = _ensure_csv(path)
    _assert_writable_schema(path)
    # Build ordered row
    ordered = [str(row.get(col, "")) for col in CSV_COLUMNS]
    buffer = io.StringIO(newline="")
    csv.writer(buffer).writerow(ordered)
    atomic_write(path, path.read_text() + buffer.getvalue())
    logger.info(
        "Saved prediction for %s at price $%s",
        row.get("timestamp_utc", "?"),
        row.get("initial_price", "?"),
    )


def update_verification(
    timestamp_utc: str,
    actual_price: float,
    absolute_error: float,
    percentage_error: float,
    actual_direction: str,
    direction_correct: bool,
    within_range: bool,
    source_coingecko: float | None = None,
    source_binance: float | None = None,
    price_confidence: str = "",
    path: Path | None = None,
) -> bool:
    """Update a prediction row with verification data.

    Finds the row by timestamp_utc and fills in the actual price and
    computed error metrics. Never deletes or reorders existing rows.

    Args:
        timestamp_utc: The UTC timestamp string to match.
        actual_price: The actual BTC/USD price after 24h.
        absolute_error: Computed absolute error.
        percentage_error: Computed percentage error.
        actual_direction: 'up', 'down', or 'flat'.
        direction_correct: Whether predicted direction matched.
        within_range: Whether actual price was in [min, max].
        source_coingecko: CoinGecko price (for cross-verification).
        source_binance: Binance price (for cross-verification).
        price_confidence: 'high', 'medium', or 'low'.
        path: Optional override for the CSV path.

    Returns:
        True if a row was updated, False if no matching row was found.
    """
    path = _ensure_csv(path)
    _assert_writable_schema(path)
    df = load_predictions(path)

    mask = df["timestamp_utc"] == timestamp_utc
    if not mask.any():
        logger.warning("No prediction found for timestamp %s", timestamp_utc)
        return False

    idx = df.index[mask][0]
    df.at[idx, "actual_price_24h"] = str(actual_price)
    df.at[idx, "absolute_error"] = str(round(absolute_error, 2))
    df.at[idx, "percentage_error"] = str(round(percentage_error, 4))
    df.at[idx, "actual_direction"] = actual_direction
    df.at[idx, "direction_correct"] = str(direction_correct).lower()
    df.at[idx, "within_range"] = str(within_range).lower()

    # Multi-source verification fields
    if source_coingecko is not None:
        df.at[idx, "source_coingecko"] = str(round(source_coingecko, 2))
    if source_binance is not None:
        df.at[idx, "source_binance"] = str(round(source_binance, 2))
    if price_confidence:
        df.at[idx, "price_confidence"] = price_confidence

    # Write back — preserve all existing data
    atomic_write(path, df.to_csv(index=False))
    logger.info(
        "Updated verification for %s: actual=$%.2f, error=%.2f%%, confidence=%s",
        timestamp_utc,
        actual_price,
        percentage_error,
        price_confidence or "n/a",
    )
    return True
