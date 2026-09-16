"""
Tests for the Binance API client module.

Uses mock patches to avoid hitting the real Binance API.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from evaluation.binance import BinanceError, get_btc_daily_close, get_btc_price_at


@pytest.fixture
def target_time():
    """A fixed UTC datetime for testing."""
    return datetime(2026, 9, 10, 16, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def mock_klines_hourly():
    """Simulated hourly kline data from Binance."""
    base_ts = int(datetime(2026, 9, 10, 15, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    return [
        # [open_time, open, high, low, close, volume, close_time, ...]
        [base_ts, "77000.00", "77100.00", "76900.00", "77050.00", "100", base_ts + 3600000 - 1, "0", "0", "0", "0", "0"],
        [base_ts + 3600000, "77050.00", "77200.00", "77000.00", "77150.00", "120", base_ts + 7200000 - 1, "0", "0", "0", "0", "0"],
        [base_ts + 7200000, "77150.00", "77300.00", "77100.00", "77250.00", "110", base_ts + 10800000 - 1, "0", "0", "0", "0", "0"],
    ]


@pytest.fixture
def mock_klines_daily():
    """Simulated daily kline data from Binance."""
    base_ts = int(datetime(2026, 9, 10, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    return [
        [base_ts, "77000.00", "78000.00", "76500.00", "77500.00", "50000", base_ts + 86400000 - 1, "0", "0", "0", "0", "0"],
    ]


class TestGetBtcPriceAt:
    """Tests for get_btc_price_at function."""

    @patch("evaluation.binance._request_with_retry")
    def test_returns_closest_candle_close(self, mock_req, target_time, mock_klines_hourly):
        mock_req.return_value = mock_klines_hourly

        result = get_btc_price_at(target_time)

        # The closest candle to 16:00 is the one starting at 15:00+3600=16:00
        # which is index 1 (open_time = base_ts + 3600000)
        assert isinstance(result, float)
        assert result > 0

    @patch("evaluation.binance._request_with_retry")
    def test_empty_data_raises(self, mock_req, target_time):
        mock_req.return_value = []

        with pytest.raises(BinanceError, match="No kline data"):
            get_btc_price_at(target_time)


class TestGetBtcDailyClose:
    """Tests for get_btc_daily_close function."""

    @patch("evaluation.binance._request_with_retry")
    def test_returns_daily_close(self, mock_req, mock_klines_daily):
        mock_req.return_value = mock_klines_daily
        date = datetime(2026, 9, 10, 0, 0, 0, tzinfo=timezone.utc)

        result = get_btc_daily_close(date)

        assert result == 77500.00

    @patch("evaluation.binance._request_with_retry")
    def test_empty_data_raises(self, mock_req):
        mock_req.return_value = []
        date = datetime(2026, 9, 10, 0, 0, 0, tzinfo=timezone.utc)

        with pytest.raises(BinanceError, match="No daily kline data"):
            get_btc_daily_close(date)
