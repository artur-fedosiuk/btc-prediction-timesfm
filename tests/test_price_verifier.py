"""
Tests for the TWAP-based price_verifier module.

Uses mock patches to simulate CoinGecko and Binance hourly
price responses without hitting real APIs.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from evaluation.price_verifier import VerifiedPrice, get_verified_price, _compute_twap


@pytest.fixture
def target_time():
    """A fixed UTC datetime for testing (end of 24h window)."""
    return datetime(2026, 9, 10, 16, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def mock_cg_hourly_prices():
    """Simulated CoinGecko hourly prices (24 points)."""
    base = datetime(2026, 9, 9, 16, 0, 0, tzinfo=timezone.utc)
    # Prices around $77,000 with some variation
    prices = [
        77000, 77050, 77100, 77080, 77120, 77200,
        77150, 77180, 77250, 77300, 77280, 77320,
        77350, 77400, 77380, 77420, 77450, 77500,
        77480, 77520, 77550, 77600, 77580, 77620,
    ]
    return [(base + timedelta(hours=i), float(p)) for i, p in enumerate(prices)]


@pytest.fixture
def mock_bn_hourly_prices():
    """Simulated Binance hourly prices (24 points, slightly different)."""
    base = datetime(2026, 9, 9, 16, 0, 0, tzinfo=timezone.utc)
    # Prices very close to CoinGecko (within 0.1%)
    prices = [
        77010, 77060, 77090, 77090, 77130, 77190,
        77160, 77170, 77260, 77290, 77290, 77310,
        77360, 77390, 77390, 77410, 77460, 77490,
        77490, 77510, 77560, 77590, 77590, 77610,
    ]
    return [(base + timedelta(hours=i), float(p)) for i, p in enumerate(prices)]


class TestComputeTwap:
    """Tests for the TWAP computation."""

    def test_basic_average(self):
        prices = [
            (datetime(2026, 1, 1, h, tzinfo=timezone.utc), float(100 + h))
            for h in range(24)
        ]
        twap = _compute_twap(prices)
        expected = sum(100 + h for h in range(24)) / 24
        assert abs(twap - expected) < 0.001

    def test_single_price(self):
        prices = [(datetime(2026, 1, 1, tzinfo=timezone.utc), 77000.0)]
        assert _compute_twap(prices) == 77000.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _compute_twap([])


class TestGetVerifiedPrice:
    """Tests for the TWAP-based multi-source price verifier."""

    @patch("evaluation.price_verifier.binance_api.get_btc_hourly_prices")
    @patch("evaluation.price_verifier.coingecko_api.get_btc_hourly_prices")
    def test_both_sources_agree_twap(
        self, mock_cg, mock_bn, target_time,
        mock_cg_hourly_prices, mock_bn_hourly_prices
    ):
        """When both TWAP sources agree within 1%, use average with high confidence."""
        mock_cg.return_value = mock_cg_hourly_prices
        mock_bn.return_value = mock_bn_hourly_prices

        result = get_verified_price(target_time)

        assert isinstance(result, VerifiedPrice)
        assert result.confidence == "high"
        assert result.source_coingecko is not None
        assert result.source_binance is not None
        assert result.num_hourly_points == 24
        assert result.discrepancy_pct is not None
        assert result.discrepancy_pct < 1.0
        # TWAP should be average of the two TWAPs
        cg_twap = sum(p for _, p in mock_cg_hourly_prices) / 24
        bn_twap = sum(p for _, p in mock_bn_hourly_prices) / 24
        expected_avg = (cg_twap + bn_twap) / 2
        assert abs(result.price - expected_avg) < 0.01

    @patch("evaluation.price_verifier.binance_api.get_btc_hourly_prices")
    @patch("evaluation.price_verifier.coingecko_api.get_btc_hourly_prices")
    def test_sources_disagree_twap(self, mock_cg, mock_bn, target_time):
        """When TWAPs disagree by >1%, use CoinGecko with low confidence."""
        base = datetime(2026, 9, 9, 16, 0, 0, tzinfo=timezone.utc)
        # CoinGecko prices around 80000
        cg_prices = [(base + timedelta(hours=i), 80000.0) for i in range(24)]
        # Binance prices around 78000 (~2.5% difference)
        bn_prices = [(base + timedelta(hours=i), 78000.0) for i in range(24)]

        mock_cg.return_value = cg_prices
        mock_bn.return_value = bn_prices

        result = get_verified_price(target_time)

        assert result.confidence == "low"
        assert result.price == 80000.0  # CoinGecko used as primary
        assert result.discrepancy_pct > 1.0

    @patch("evaluation.price_verifier.binance_api.get_btc_hourly_prices")
    @patch("evaluation.price_verifier.coingecko_api.get_btc_hourly_prices")
    def test_only_coingecko_available(
        self, mock_cg, mock_bn, target_time, mock_cg_hourly_prices
    ):
        """When only CoinGecko responds, medium confidence."""
        mock_cg.return_value = mock_cg_hourly_prices
        mock_bn.side_effect = Exception("Binance down")

        result = get_verified_price(target_time)

        assert result.confidence == "medium"
        assert result.source_coingecko is not None
        assert result.source_binance is None
        assert result.num_hourly_points == 24

    @patch("evaluation.price_verifier.binance_api.get_btc_hourly_prices")
    @patch("evaluation.price_verifier.coingecko_api.get_btc_hourly_prices")
    def test_only_binance_available(
        self, mock_cg, mock_bn, target_time, mock_bn_hourly_prices
    ):
        """When only Binance responds, medium confidence."""
        mock_cg.side_effect = Exception("CoinGecko down")
        mock_bn.return_value = mock_bn_hourly_prices

        result = get_verified_price(target_time)

        assert result.confidence == "medium"
        assert result.source_coingecko is None
        assert result.source_binance is not None

    @patch("evaluation.price_verifier.binance_api.get_btc_hourly_prices")
    @patch("evaluation.price_verifier.coingecko_api.get_btc_hourly_prices")
    def test_both_fail_raises(self, mock_cg, mock_bn, target_time):
        """When both sources fail, raise RuntimeError."""
        mock_cg.side_effect = Exception("CoinGecko down")
        mock_bn.side_effect = Exception("Binance down")

        with pytest.raises(RuntimeError, match="Both CoinGecko and Binance failed"):
            get_verified_price(target_time)

    @patch("evaluation.price_verifier.binance_api.get_btc_hourly_prices")
    @patch("evaluation.price_verifier.coingecko_api.get_btc_hourly_prices")
    def test_high_low_tracked(
        self, mock_cg, mock_bn, target_time,
        mock_cg_hourly_prices, mock_bn_hourly_prices
    ):
        """High and low prices are tracked from hourly data."""
        mock_cg.return_value = mock_cg_hourly_prices
        mock_bn.return_value = mock_bn_hourly_prices

        result = get_verified_price(target_time)

        assert result.price_high is not None
        assert result.price_low is not None
        assert result.price_high >= result.price_low
        # High should be the max of all hourly prices from both sources
        all_prices = [p for _, p in mock_cg_hourly_prices] + [p for _, p in mock_bn_hourly_prices]
        assert result.price_high == max(all_prices)
        assert result.price_low == min(all_prices)


class TestVerifiedPriceDataclass:
    """Tests for the VerifiedPrice dataclass."""

    def test_all_fields_set(self):
        vp = VerifiedPrice(
            price=77300.0,
            source_coingecko=77300.0,
            source_binance=77290.0,
            confidence="high",
            discrepancy_pct=0.013,
            num_hourly_points=24,
            price_high=77620.0,
            price_low=77000.0,
        )
        assert vp.price == 77300.0
        assert vp.confidence == "high"
        assert vp.num_hourly_points == 24

    def test_none_sources(self):
        vp = VerifiedPrice(
            price=77300.0,
            source_coingecko=None,
            source_binance=None,
            confidence="low",
            discrepancy_pct=None,
        )
        assert vp.source_coingecko is None
        assert vp.source_binance is None
        assert vp.num_hourly_points == 0
