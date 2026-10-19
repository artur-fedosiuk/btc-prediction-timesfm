"""Tests for scripts/trend_baseline.py.

Verifies:
1. A hand-computed 12-day series with SMA(3) and known trades.
2. No-look-ahead test: changing prices after day t never changes the position at t.
3. A no-trade edge case.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.ohlc_io import Candle, save_csv
from scripts.trend_baseline import (
    CostModel,
    FeeScenario,
    compute_signals,
    compute_sma,
    extract_trades,
    evaluate_baseline,
    get_baseline_scenarios,
    run_baseline,
)


# ---------------------------------------------------------------------------
# Test 1: Hand-computed 12-day series with SMA(3) and known trades
# ---------------------------------------------------------------------------


def test_hand_computed_twelve_day_sma3() -> None:
    """A 12-day hand-computed price series with SMA(3).

    Prices:
      Day 0: 100.0
      Day 1: 102.0
      Day 2: 105.0
      Day 3: 103.0
      Day 4: 107.0
      Day 5: 106.0
      Day 6: 108.0
      Day 7: 110.0
      Day 8: 109.0
      Day 9: 112.0
      Day 10: 111.0
      Day 11: 115.0

    SMA(3) values (starting from day 2):
      Day 2: (100+102+105)/3 = 102.333
      Day 3: (102+105+103)/3 = 103.333
      Day 4: (105+103+107)/3 = 105.0
      Day 5: (103+107+106)/3 = 105.333
      Day 6: (107+106+108)/3 = 107.0
      Day 7: (106+108+110)/3 = 108.0
      Day 8: (108+110+109)/3 = 109.0
      Day 9: (110+109+112)/3 = 110.333
      Day 10: (109+112+111)/3 = 110.667
      Day 11: (112+111+115)/3 = 112.667

    Signals (close > SMA):
      Day 0: False (no SMA)
      Day 1: False (no SMA)
      Day 2: 105 > 102.333 -> True
      Day 3: 103 > 103.333 -> False
      Day 4: 107 > 105.0 -> True
      Day 5: 106 > 105.333 -> True
      Day 6: 108 > 107.0 -> True
      Day 7: 110 > 108.0 -> True
      Day 8: 109 > 109.0 -> False
      Day 9: 112 > 110.333 -> True
      Day 10: 111 > 110.667 -> True
      Day 11: 115 > 112.667 -> True

    Trades (runs of True):
      Trade 1: Entry day 2 (105), Exit day 3 (103) -> r = 103/105 - 1 = -0.0190476
      Trade 2: Entry day 4 (107), Exit day 8 (109) -> r = 109/107 - 1 = 0.0186916
      Trade 3: Entry day 9 (112), Exit day 11 (115) -> r = 115/112 - 1 = 0.0267857
    """
    prices = [100.0, 102.0, 105.0, 103.0, 107.0, 106.0, 108.0, 110.0, 109.0, 112.0, 111.0, 115.0]

    # Test SMA
    sma = compute_sma(prices, 3)
    assert sma[0] is None
    assert sma[1] is None
    assert sma[2] == (100 + 102 + 105) / 3
    assert sma[3] == (102 + 105 + 103) / 3
    assert sma[11] == (112 + 111 + 115) / 3

    # Test signals
    signals = compute_signals(prices, 3)
    expected_signals = [False, False, True, False, True, True, True, True, False, True, True, True]
    assert signals == expected_signals

    # Test trades
    trades = extract_trades(prices, signals)
    assert len(trades) == 3

    # Trade 1: entry at day 2 (105), exit at day 3 (103)
    assert trades[0].entry_idx == 2
    assert trades[0].exit_idx == 3
    assert trades[0].entry_price == 105.0
    assert trades[0].exit_price == 103.0
    assert abs(trades[0].gross_return - (103.0 / 105.0 - 1.0)) < 1e-10

    # Trade 2: entry at day 4 (107), exit at day 8 (109)
    assert trades[1].entry_idx == 4
    assert trades[1].exit_idx == 8
    assert trades[1].entry_price == 107.0
    assert trades[1].exit_price == 109.0
    assert abs(trades[1].gross_return - (109.0 / 107.0 - 1.0)) < 1e-10

    # Trade 3: entry at day 9 (112), exit at day 11 (115)
    assert trades[2].entry_idx == 9
    assert trades[2].exit_idx == 11
    assert trades[2].entry_price == 112.0
    assert trades[2].exit_price == 115.0
    assert abs(trades[2].gross_return - (115.0 / 112.0 - 1.0)) < 1e-10


# ---------------------------------------------------------------------------
# Test 2: No-look-ahead property
# ---------------------------------------------------------------------------


def test_no_look_ahead() -> None:
    """Changing prices after day t never changes the position at day t."""
    # Base prices
    base_prices = [100.0, 102.0, 105.0, 103.0, 107.0, 106.0, 108.0, 110.0, 109.0, 112.0, 111.0, 115.0]

    # Get signals for base prices
    base_signals = compute_signals(base_prices, 3)

    # Modify prices after day 5 (should not affect signals at days 0-5)
    modified_prices = base_prices.copy()
    for i in range(6, len(modified_prices)):
        modified_prices[i] = modified_prices[i] * 1.5  # Drastically change

    modified_signals = compute_signals(modified_prices, 3)

    # Signals up to day 5 should be identical
    for i in range(6):
        assert base_signals[i] == modified_signals[i], f"Signal at day {i} changed due to look-ahead!"


# ---------------------------------------------------------------------------
# Test 3: No-trade edge case
# ---------------------------------------------------------------------------


def test_no_trade_edge_case() -> None:
    """When price never exceeds SMA, no trades should be generated."""
    # Monotonically decreasing prices
    prices = [100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 91.0]

    signals = compute_signals(prices, 3)
    assert all(s is False for s in signals)

    trades = extract_trades(prices, signals)
    assert len(trades) == 0

    # Evaluate with no trades
    model = CostModel(f=0.0025, e=0.0015)
    scenario = FeeScenario("test", f=0.0025, e=0.0015)
    result = evaluate_baseline(prices, signals, model, scenario, 3)

    assert result.trades_count == 0
    assert result.hit_rate == 0.0
    assert result.total_net_after_tax == 0.0
    assert result.max_drawdown == 0.0


# ---------------------------------------------------------------------------
# Test 4: CostModel properties and tax calculation
# ---------------------------------------------------------------------------


def test_cost_model_properties() -> None:
    """Test CostModel properties and tax calculation."""
    model = CostModel(f=0.0025, e=0.0015, tax_rate=0.33)

    # M = (1-0.0025)^2 * (1-0.0015) = 0.99500625 * 0.9985 = 0.993515...
    expected_M = (1.0 - 0.0025) ** 2 * (1.0 - 0.0015)
    assert abs(model.M - expected_M) < 1e-12

    # c = 1 - M
    assert abs(model.c - (1.0 - expected_M)) < 1e-12

    # net_cash for r=0.02
    net = model.net_cash(0.02)
    expected_net = expected_M * 1.02 - 1.0
    assert abs(net - expected_net) < 1e-12

    # net_after_tax for r=0.02
    net_tax = model.net_after_tax(0.02)
    expected_tax = 0.33 * 0.02
    expected_net_tax = expected_net - expected_tax
    assert abs(net_tax - expected_net_tax) < 1e-12


# ---------------------------------------------------------------------------
# Test 5: Integration test - run_baseline on synthetic CSV
# ---------------------------------------------------------------------------


def test_integration_run_baseline_synthetic(tmp_path: Path) -> None:
    """Integration test: run_baseline on a small synthetic CSV (60 days, valid OHLC)."""
    import json

    # Generate 60 days of valid OHLC data with known pattern
    from datetime import date, timedelta

    # Create prices that go up and down
    base = 50000.0
    prices = []
    for i in range(60):
        if i < 20:
            base *= 1.01  # uptrend
        elif i < 40:
            base *= 0.99  # downtrend
        else:
            base *= 1.005  # slight uptrend
        prices.append(base)

    # Build candles
    start_date = date(2023, 1, 1)
    candles = []
    for i, close in enumerate(prices):
        d = start_date + timedelta(days=i)
        high = close * 1.02
        low = close * 0.98
        open_ = close * 0.99
        candles.append(Candle(d, open_, high, low, close, 100.0))

    # Write CSV using save_csv (correct format)
    csv_path = tmp_path / "btc_usd_1d.csv"
    save_csv(candles, csv_path)

    # Write meta
    meta_path = tmp_path / "btc_usd_1d.meta.json"
    meta_path.write_text(json.dumps({
        "source": "synthetic_test",
        "product": "BTC-USD",
        "granularity": 86400,
        "fetched_at_utc": "2024-01-01T00:00:00Z",
        "rows": 60,
        "first_date": "2023-01-01",
        "last_date": "2023-03-01",
        "sha256": "test"
    }))

    # Run baseline
    results, meta, years = run_baseline(csv_path)

    # Verify structure
    assert len(results) > 0
    # Should have 3 SMA windows × 2 scenarios = 6 results
    assert len(results) == 6
    assert years > 0

    # Check each result has the right structure
    for res in results:
        assert res.sma_window in {50, 100, 200}
        assert res.scenario.name in ["coinbase_eu_maker", "coinbase_eu_taker"]
        assert res.cost_model.f > 0
        assert res.cost_model.e >= 0
        assert res.cost_model.tax_rate == 0.33
        assert res.trades_count >= 0
        assert 0 <= res.hit_rate <= 1
        assert res.max_drawdown >= 0
        assert len(res.equity_curve) == res.trades_count + 1
        assert res.equity_curve[0] == 1.0


# ---------------------------------------------------------------------------
# Test 6: get_baseline_scenarios returns correct scenarios
# ---------------------------------------------------------------------------


def test_get_baseline_scenarios() -> None:
    """Test that get_baseline_scenarios returns the two correct scenarios."""
    scenarios = get_baseline_scenarios(0.0015)
    assert len(scenarios) == 2
    assert scenarios[0].name == "coinbase_eu_maker"
    assert scenarios[0].f == 0.0025
    assert scenarios[0].e == 0.0015
    assert scenarios[1].name == "coinbase_eu_taker"
    assert scenarios[1].f == 0.0050
    assert scenarios[1].e == 0.0015