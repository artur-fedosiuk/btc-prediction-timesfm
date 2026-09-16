"""
Binance public API client for BTC/USDT price data.

Used as a second source to cross-verify CoinGecko prices.
No API key required — uses the public REST endpoint.

Features:
- Retry with back-off on transient failures (429, 5xx).
- All timestamps returned as UTC-aware datetimes.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import requests

from .config import API_MAX_RETRIES, API_RETRY_DELAYS

logger = logging.getLogger(__name__)

BINANCE_BASE_URL = "https://api.binance.com"


class BinanceError(Exception):
    """Raised when Binance API calls fail after all retries."""


def _request_with_retry(url: str, params: dict | None = None) -> dict | list:
    """GET request with retry logic for transient errors."""
    last_exception: Exception | None = None
    for attempt in range(API_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 418):
                # Rate-limited
                delay = API_RETRY_DELAYS[
                    min(attempt, len(API_RETRY_DELAYS) - 1)
                ]
                logger.warning(
                    "Binance rate limit (%d). Retrying in %ds "
                    "(attempt %d/%d)",
                    resp.status_code,
                    delay,
                    attempt + 1,
                    API_MAX_RETRIES,
                )
                time.sleep(delay)
                continue
            if resp.status_code >= 500:
                delay = API_RETRY_DELAYS[
                    min(attempt, len(API_RETRY_DELAYS) - 1)
                ]
                logger.warning(
                    "Binance server error (%d). Retrying in %ds "
                    "(attempt %d/%d)",
                    resp.status_code,
                    delay,
                    attempt + 1,
                    API_MAX_RETRIES,
                )
                time.sleep(delay)
                continue
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            last_exception = exc
            delay = API_RETRY_DELAYS[
                min(attempt, len(API_RETRY_DELAYS) - 1)
            ]
            logger.warning(
                "Binance request failed: %s. Retrying in %ds "
                "(attempt %d/%d)",
                exc,
                delay,
                attempt + 1,
                API_MAX_RETRIES,
            )
            time.sleep(delay)
    raise BinanceError(
        f"Binance API failed after {API_MAX_RETRIES} retries. "
        f"Last error: {last_exception}"
    )


def get_btc_price_at(target: datetime) -> float:
    """Get BTC/USDT price closest to a specific UTC timestamp.

    Uses the Binance klines (candlestick) endpoint with 1-hour interval
    to find the close price of the candle closest to `target`.

    Args:
        target: Target UTC datetime.

    Returns:
        BTC/USDT close price of the hourly candle closest to the target.

    Raises:
        BinanceError: If the API call fails or no data is available.
    """
    target_ms = int(target.timestamp() * 1000)
    # Fetch a ±6h window of 1-hour candles around the target
    start_ms = target_ms - 6 * 3600 * 1000
    end_ms = target_ms + 6 * 3600 * 1000

    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    data = _request_with_retry(
        url,
        params={
            "symbol": "BTCUSDT",
            "interval": "1h",
            "startTime": str(start_ms),
            "endTime": str(end_ms),
            "limit": "24",
        },
    )

    if not data:
        raise BinanceError(
            f"No kline data available around {target.isoformat()}"
        )

    # Each kline: [open_time, open, high, low, close, volume, close_time, ...]
    # Find the candle whose open_time is closest to target
    closest = min(data, key=lambda k: abs(int(k[0]) - target_ms))
    close_price = float(closest[4])  # close price

    candle_time = datetime.fromtimestamp(
        int(closest[0]) / 1000.0, tz=timezone.utc
    )
    logger.info(
        "Binance BTC/USDT at %s: $%.2f (candle %s)",
        target.isoformat(),
        close_price,
        candle_time.isoformat(),
    )
    return close_price


def get_btc_daily_close(date: datetime) -> float:
    """Get the BTC/USDT daily close price for a specific date.

    Uses the Binance klines endpoint with 1-day interval.
    The daily candle closes at 00:00 UTC the following day.

    Args:
        date: UTC date to get the daily close for.

    Returns:
        BTC/USDT daily close price.

    Raises:
        BinanceError: If the API call fails or no data is available.
    """
    # Binance daily candle opens at 00:00 UTC
    date_start = date.replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start_ms = int(date_start.timestamp() * 1000)
    # Fetch just 1 day
    end_ms = start_ms + 24 * 3600 * 1000

    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    data = _request_with_retry(
        url,
        params={
            "symbol": "BTCUSDT",
            "interval": "1d",
            "startTime": str(start_ms),
            "endTime": str(end_ms),
            "limit": "1",
        },
    )

    if not data:
        raise BinanceError(
            f"No daily kline data for {date.strftime('%Y-%m-%d')}"
        )

    close_price = float(data[0][4])
    logger.info(
        "Binance BTC/USDT daily close %s: $%.2f",
        date.strftime("%Y-%m-%d"),
        close_price,
    )
    return close_price


def get_btc_hourly_prices(
    start: datetime, hours: int = 24
) -> list[tuple[datetime, float]]:
    """Fetch BTC/USDT hourly close prices for a time window.

    Uses the Binance klines endpoint with 1-hour interval to retrieve
    all hourly candle close prices in the specified window.

    Args:
        start: Start of the window (UTC).
        hours: Number of hours to fetch (default 24).

    Returns:
        List of (datetime_utc, close_price_usd) tuples sorted chronologically.

    Raises:
        BinanceError: If the API call fails or no data is available.
    """
    start_ms = int(start.timestamp() * 1000)
    end_ms = start_ms + hours * 3600 * 1000

    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    data = _request_with_retry(
        url,
        params={
            "symbol": "BTCUSDT",
            "interval": "1h",
            "startTime": str(start_ms),
            "endTime": str(end_ms),
            "limit": str(min(hours + 1, 1000)),
        },
    )

    if not data:
        raise BinanceError(
            f"No hourly kline data from {start.isoformat()} for {hours}h"
        )

    result = []
    for kline in data:
        dt = datetime.fromtimestamp(
            int(kline[0]) / 1000.0, tz=timezone.utc
        )
        close_price = float(kline[4])
        result.append((dt, close_price))

    logger.info(
        "Binance fetched %d hourly prices from %s",
        len(result),
        start.isoformat(),
    )
    return sorted(result, key=lambda x: x[0])

