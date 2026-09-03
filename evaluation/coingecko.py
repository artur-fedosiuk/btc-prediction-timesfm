"""
CoinGecko API client for BTC/USD price data.

Features:
- Retry with exponential backoff on transient failures (429, 5xx).
- Optional API key via COINGECKO_API_KEY env var.
- All timestamps returned as UTC-aware datetimes.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import requests

from .config import (
    API_MAX_RETRIES,
    API_RETRY_DELAYS,
    COINGECKO_API_KEY,
    COINGECKO_BASE_URL,
)

logger = logging.getLogger(__name__)


class CoinGeckoError(Exception):
    """Raised when CoinGecko API calls fail after all retries."""


def _headers() -> dict[str, str]:
    """Build request headers, including the API key if configured."""
    headers = {"Accept": "application/json"}
    if COINGECKO_API_KEY:
        headers["x-cg-demo-api-key"] = COINGECKO_API_KEY
    return headers


def _request_with_retry(url: str, params: dict | None = None) -> dict:
    """GET request with retry logic for transient errors."""
    last_exception: Exception | None = None
    for attempt in range(API_MAX_RETRIES):
        try:
            resp = requests.get(
                url, params=params, headers=_headers(), timeout=30
            )
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                # Rate-limited — wait and retry
                delay = API_RETRY_DELAYS[
                    min(attempt, len(API_RETRY_DELAYS) - 1)
                ]
                logger.warning(
                    "CoinGecko rate limit (429). Retrying in %ds "
                    "(attempt %d/%d)",
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
                    "CoinGecko server error (%d). Retrying in %ds "
                    "(attempt %d/%d)",
                    resp.status_code,
                    delay,
                    attempt + 1,
                    API_MAX_RETRIES,
                )
                time.sleep(delay)
                continue
            # Client error (4xx other than 429) — don't retry
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            last_exception = exc
            delay = API_RETRY_DELAYS[
                min(attempt, len(API_RETRY_DELAYS) - 1)
            ]
            logger.warning(
                "CoinGecko request failed: %s. Retrying in %ds "
                "(attempt %d/%d)",
                exc,
                delay,
                attempt + 1,
                API_MAX_RETRIES,
            )
            time.sleep(delay)
    raise CoinGeckoError(
        f"CoinGecko API failed after {API_MAX_RETRIES} retries. "
        f"Last error: {last_exception}"
    )


def get_current_btc_price() -> float:
    """Fetch the current BTC/USD price.

    Returns:
        Current Bitcoin price in USD.

    Raises:
        CoinGeckoError: If the API call fails after retries.
    """
    url = f"{COINGECKO_BASE_URL}/simple/price"
    data = _request_with_retry(
        url, params={"ids": "bitcoin", "vs_currencies": "usd"}
    )
    try:
        return float(data["bitcoin"]["usd"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CoinGeckoError(
            f"Unexpected response format: {data}"
        ) from exc


def get_btc_history(days: int = 90) -> list[tuple[datetime, float]]:
    """Fetch BTC/USD historical prices (hourly granularity).

    Args:
        days: Number of days of history to fetch (max 90 for hourly).

    Returns:
        List of (datetime_utc, price_usd) tuples sorted chronologically.

    Raises:
        CoinGeckoError: If the API call fails after retries.
    """
    url = f"{COINGECKO_BASE_URL}/coins/bitcoin/market_chart"
    data = _request_with_retry(
        url, params={"vs_currency": "usd", "days": str(days)}
    )
    try:
        prices = data["prices"]
        result = []
        for timestamp_ms, price in prices:
            dt = datetime.fromtimestamp(
                timestamp_ms / 1000.0, tz=timezone.utc
            )
            result.append((dt, float(price)))
        return sorted(result, key=lambda x: x[0])
    except (KeyError, TypeError, ValueError) as exc:
        raise CoinGeckoError(
            f"Unexpected history response format: {data}"
        ) from exc


def get_btc_price_at(target: datetime) -> float:
    """Get the BTC/USD price closest to a specific UTC timestamp.

    Fetches a ±2-day window around the target and returns the price
    at the timestamp closest to `target`.

    Args:
        target: Target UTC datetime.

    Returns:
        BTC/USD price closest to the target timestamp.

    Raises:
        CoinGeckoError: If the API call fails or no data is available.
    """
    # CoinGecko /coins/bitcoin/market_chart/range accepts UNIX timestamps
    from_ts = int(target.timestamp()) - 86400  # 1 day before
    to_ts = int(target.timestamp()) + 86400  # 1 day after

    url = f"{COINGECKO_BASE_URL}/coins/bitcoin/market_chart/range"
    data = _request_with_retry(
        url,
        params={
            "vs_currency": "usd",
            "from": str(from_ts),
            "to": str(to_ts),
        },
    )
    try:
        prices = data["prices"]
        if not prices:
            raise CoinGeckoError(
                f"No price data available around {target.isoformat()}"
            )

        target_ts = target.timestamp()
        closest = min(prices, key=lambda p: abs(p[0] / 1000.0 - target_ts))
        return float(closest[1])
    except (KeyError, TypeError, ValueError) as exc:
        raise CoinGeckoError(
            f"Unexpected range response format: {data}"
        ) from exc
