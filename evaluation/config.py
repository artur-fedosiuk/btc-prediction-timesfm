"""
Configuration for the Bitcoin prediction evaluation experiment.

All timestamps use UTC. The experiment runs for EXPERIMENT_DURATION_DAYS
starting from EXPERIMENT_START_DATE.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to repository root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "data" / "bitcoin_predictions.csv"
REPORT_PATH = REPO_ROOT / "reports" / "bitcoin_report.md"

# ---------------------------------------------------------------------------
# Experiment window
# ---------------------------------------------------------------------------
EXPERIMENT_START_DATE: str = os.environ.get(
    "EXPERIMENT_START_DATE", "2026-09-16"
)
EXPERIMENT_DURATION_DAYS: int = int(
    os.environ.get("EXPERIMENT_DURATION_DAYS", "30")
)

# ---------------------------------------------------------------------------
# CoinGecko API
# ---------------------------------------------------------------------------
COINGECKO_BASE_URL: str = os.environ.get(
    "COINGECKO_BASE_URL", "https://api.coingecko.com/api/v3"
)
COINGECKO_API_KEY: str | None = os.environ.get("COINGECKO_API_KEY")

# Retry configuration
API_MAX_RETRIES: int = 3
API_RETRY_DELAYS: list[int] = [10, 30, 60]  # seconds between retries

# ---------------------------------------------------------------------------
# CSV columns
# ---------------------------------------------------------------------------
CSV_COLUMNS: list[str] = [
    "timestamp_utc",
    "experiment_version",  # new in 1.5
    "forecast_horizon",    # new in 1.5
    "initial_price",
    "predicted_price_24h",
    "predicted_min",
    "predicted_max",
    "actual_price_24h",
    "absolute_error",
    "percentage_error",
    "predicted_direction",
    "actual_direction",
    "direction_correct",
    "within_range",
    "prediction_method",
    "source_coingecko",
    "source_binance",
    "price_confidence",
    
    # Path metrics (added in 1.5)
    "pred_t1", "pred_t4", "pred_t8", "pred_t12", "pred_t24",
    "pred_return_t1_pct", "pred_return_t4_pct", "pred_return_t8_pct", "pred_return_t12_pct", "pred_return_t24_pct",
    "pred_path_min", "pred_path_max",
    "pred_min_return_pct", "pred_max_return_pct",
    "pred_path_range_pct", "pred_path_slope", "pred_path_volatility",
    "forecast_path", "forecast_lower", "forecast_upper",
]


def experiment_start() -> datetime:
    """Return the experiment start as a timezone-aware UTC datetime."""
    return datetime.strptime(EXPERIMENT_START_DATE, "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )


def experiment_end() -> datetime:
    """Return the experiment end as a timezone-aware UTC datetime."""
    from datetime import timedelta

    return experiment_start() + timedelta(days=EXPERIMENT_DURATION_DAYS)


def is_experiment_active() -> bool:
    """Check whether the current UTC time is within the experiment window."""
    now = datetime.now(timezone.utc)
    return experiment_start() <= now <= experiment_end()
