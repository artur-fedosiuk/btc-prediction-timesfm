"""
Multi-source price verifier for BTC/USD using TWAP.

Cross-references CoinGecko and Binance to produce a high-confidence
actual price using TWAP (Time-Weighted Average Price) — the simple
average of all hourly prices over a 24-hour window.

Both sources are recorded so the user can inspect discrepancies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from . import binance as binance_api
from . import coingecko as coingecko_api

logger = logging.getLogger(__name__)

# Maximum acceptable discrepancy between sources (percentage)
MAX_DISCREPANCY_PCT = 1.0


@dataclass
class VerifiedPrice:
    """Result of multi-source price verification."""

    price: float  # Final TWAP price used for evaluation
    source_coingecko: float | None  # CoinGecko TWAP (None if failed)
    source_binance: float | None  # Binance TWAP (None if failed)
    confidence: str  # "high" (both agree), "medium" (one source), "low" (mismatch)
    discrepancy_pct: float | None  # Percentage difference between sources
    num_hourly_points: int = 0  # Number of hourly prices used for TWAP
    price_high: float | None = None  # Highest hourly price in the window
    price_low: float | None = None  # Lowest hourly price in the window


def _compute_twap(prices: list[tuple[datetime, float]]) -> float:
    """Compute TWAP (simple average) from hourly prices.

    Args:
        prices: List of (datetime, price) tuples.

    Returns:
        The simple average of all prices.
    """
    if not prices:
        raise ValueError("Cannot compute TWAP from empty price list")
    return sum(p for _, p in prices) / len(prices)


def _compute_high_low(
    cg_prices: list[tuple[datetime, float]] | None,
    bn_prices: list[tuple[datetime, float]] | None,
) -> tuple[float | None, float | None]:
    """Compute the high and low from available hourly prices."""
    all_prices: list[float] = []
    if cg_prices:
        all_prices.extend(p for _, p in cg_prices)
    if bn_prices:
        all_prices.extend(p for _, p in bn_prices)
    if not all_prices:
        return None, None
    return max(all_prices), min(all_prices)


def get_verified_price(target: datetime) -> VerifiedPrice:
    """Fetch and cross-verify the BTC TWAP over a 24h window.

    For each source, fetches all hourly prices in the 24h window
    ending at `target`, computes the TWAP, and cross-verifies.

    Logic:
    - If both sources respond and TWAP agrees (within MAX_DISCREPANCY_PCT):
      use the average TWAP → confidence="high"
    - If both respond but disagree:
      use CoinGecko TWAP as primary → confidence="low", log warning
    - If only one responds:
      use that source's TWAP → confidence="medium"
    - If neither responds:
      raise an exception

    Args:
        target: UTC datetime — end of the 24h TWAP window.
                Typically prediction_time + 24h.

    Returns:
        VerifiedPrice with the final TWAP and both source values.

    Raises:
        RuntimeError: If neither source could provide hourly prices.
    """
    from datetime import timedelta

    # The 24h window: from (target - 24h) to target
    window_start = target - timedelta(hours=24)

    cg_twap: float | None = None
    bn_twap: float | None = None
    cg_prices: list[tuple[datetime, float]] | None = None
    bn_prices: list[tuple[datetime, float]] | None = None
    total_points = 0

    # --- CoinGecko hourly prices ---
    try:
        cg_prices = coingecko_api.get_btc_hourly_prices(window_start, hours=24)
        cg_twap = _compute_twap(cg_prices)
        logger.info(
            "CoinGecko TWAP (%d points, %s → %s): $%.2f",
            len(cg_prices),
            window_start.isoformat(),
            target.isoformat(),
            cg_twap,
        )
        total_points = max(total_points, len(cg_prices))
    except Exception as exc:
        logger.warning(
            "CoinGecko hourly prices failed for %s: %s",
            target.isoformat(),
            exc,
        )

    # --- Binance hourly prices ---
    try:
        bn_prices = binance_api.get_btc_hourly_prices(window_start, hours=24)
        bn_twap = _compute_twap(bn_prices)
        logger.info(
            "Binance TWAP (%d points, %s → %s): $%.2f",
            len(bn_prices),
            window_start.isoformat(),
            target.isoformat(),
            bn_twap,
        )
        total_points = max(total_points, len(bn_prices))
    except Exception as exc:
        logger.warning(
            "Binance hourly prices failed for %s: %s",
            target.isoformat(),
            exc,
        )

    # Compute high/low from all available hourly data
    price_high, price_low = _compute_high_low(cg_prices, bn_prices)

    # --- Cross-verify TWAPs ---
    if cg_twap is not None and bn_twap is not None:
        avg_twap = (cg_twap + bn_twap) / 2.0
        discrepancy = abs(cg_twap - bn_twap) / avg_twap * 100.0

        if discrepancy <= MAX_DISCREPANCY_PCT:
            logger.info(
                "TWAP sources agree (%.2f%% discrepancy). "
                "Using average: $%.2f",
                discrepancy,
                avg_twap,
            )
            return VerifiedPrice(
                price=avg_twap,
                source_coingecko=cg_twap,
                source_binance=bn_twap,
                confidence="high",
                discrepancy_pct=discrepancy,
                num_hourly_points=total_points,
                price_high=price_high,
                price_low=price_low,
            )
        else:
            logger.warning(
                "⚠️ TWAP discrepancy %.2f%% exceeds threshold (%.1f%%). "
                "CoinGecko=$%.2f, Binance=$%.2f. Using CoinGecko.",
                discrepancy,
                MAX_DISCREPANCY_PCT,
                cg_twap,
                bn_twap,
            )
            return VerifiedPrice(
                price=cg_twap,
                source_coingecko=cg_twap,
                source_binance=bn_twap,
                confidence="low",
                discrepancy_pct=discrepancy,
                num_hourly_points=total_points,
                price_high=price_high,
                price_low=price_low,
            )

    if cg_twap is not None:
        logger.info(
            "Only CoinGecko TWAP available: $%.2f (medium confidence)",
            cg_twap,
        )
        return VerifiedPrice(
            price=cg_twap,
            source_coingecko=cg_twap,
            source_binance=None,
            confidence="medium",
            discrepancy_pct=None,
            num_hourly_points=total_points,
            price_high=price_high,
            price_low=price_low,
        )

    if bn_twap is not None:
        logger.info(
            "Only Binance TWAP available: $%.2f (medium confidence)",
            bn_twap,
        )
        return VerifiedPrice(
            price=bn_twap,
            source_coingecko=None,
            source_binance=bn_twap,
            confidence="medium",
            discrepancy_pct=None,
            num_hourly_points=total_points,
            price_high=price_high,
            price_low=price_low,
        )

    raise RuntimeError(
        f"Both CoinGecko and Binance failed for {target.isoformat()}. "
        "Cannot determine actual TWAP price."
    )
