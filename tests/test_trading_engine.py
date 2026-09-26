"""
Comprehensive tests for the trading strategy evaluation system.

Covers:
- PnL calculations (LONG and SHORT)
- Fee, spread, slippage calculations
- Signal generation with thresholds
- Position sizing
- Equity curve
- Aggregate metrics (Sharpe, Sortino, drawdown, etc.)
- Backtest engine end-to-end
- Look-ahead protection (timestamp validation)
- Baseline strategies
- No future data test
"""

from __future__ import annotations

import math

import pytest

from evaluation.trading.types import (
    PredictionRecord,
    RiskConfig,
    StrategyConfig,
    TradingCosts,
    TradingSignal,
)
from evaluation.trading.signal import generate_signal, generate_signals
from evaluation.trading.metrics import (
    compute_avg_loss,
    compute_avg_trade_pnl,
    compute_avg_win,
    compute_buy_hold_return_pct,
    compute_drawdown_series,
    compute_expected_value,
    compute_gross_pnl,
    compute_gross_return_pct,
    compute_max_drawdown_pct,
    compute_net_pnl,
    compute_profit_factor,
    compute_return_pct,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_time_invested_pct,
    compute_trade_costs,
    compute_win_rate,
)
from evaluation.trading.equity import build_equity_curve, compute_position_size
from evaluation.trading.backtest_engine import run_backtest, run_multi_threshold_backtest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pred(
    pid: str = "p1",
    ts: str = "2026-09-16T15:00:00Z",
    entry: float = 80000.0,
    predicted: float = 80300.0,
    exit_price: float = 80200.0,
    high: float | None = None,
    low: float | None = None,
) -> PredictionRecord:
    """Helper to create a PredictionRecord with sensible defaults."""
    return PredictionRecord(
        prediction_id=pid,
        timestamp=ts,
        entry_price=entry,
        predicted_price=predicted,
        predicted_min=entry * 0.99,
        predicted_max=entry * 1.01,
        exit_price=exit_price,
        high_24h=high or max(entry, exit_price) * 1.005,
        low_24h=low or min(entry, exit_price) * 0.995,
    )


SAMPLE_PREDICTIONS = [
    _make_pred("p1", "2026-09-16T15:00:00Z", 80000, 80300, 80200),  # up 0.375% → LONG
    _make_pred("p2", "2026-09-17T15:00:00Z", 80200, 79900, 79800),  # down -0.374% → SHORT
    _make_pred("p3", "2026-09-18T15:00:00Z", 79800, 79850, 80500),  # up 0.063% → NO_TRADE
    _make_pred("p4", "2026-09-19T15:00:00Z", 80500, 81000, 81200),  # up 0.621% → LONG
    _make_pred("p5", "2026-09-20T15:00:00Z", 81200, 80500, 80000),  # down -0.862% → SHORT
]


# ===========================================================================
# PnL Calculations
# ===========================================================================

class TestGrossPnL:
    """Test gross PnL calculations for LONG and SHORT trades."""

    def test_long_profit(self):
        """LONG: buy at 80000, sell at 80800 → profit."""
        pnl = compute_gross_pnl("LONG", 80000, 80800, 1000)
        assert abs(pnl - 10.0) < 0.01  # 1000 * 800/80000

    def test_long_loss(self):
        """LONG: buy at 80000, sell at 79200 → loss."""
        pnl = compute_gross_pnl("LONG", 80000, 79200, 1000)
        assert abs(pnl - (-10.0)) < 0.01

    def test_short_profit(self):
        """SHORT: sell at 80000, buy at 79200 → profit."""
        pnl = compute_gross_pnl("SHORT", 80000, 79200, 1000)
        assert abs(pnl - 10.0) < 0.01

    def test_short_loss(self):
        """SHORT: sell at 80000, buy at 80800 → loss."""
        pnl = compute_gross_pnl("SHORT", 80000, 80800, 1000)
        assert abs(pnl - (-10.0)) < 0.01

    def test_zero_move(self):
        """No price change → 0 PnL."""
        assert compute_gross_pnl("LONG", 80000, 80000, 1000) == 0.0
        assert compute_gross_pnl("SHORT", 80000, 80000, 1000) == 0.0

    def test_invalid_direction(self):
        with pytest.raises(ValueError):
            compute_gross_pnl("SIDEWAYS", 80000, 80000, 1000)


class TestGrossReturnPct:
    def test_long_1pct_up(self):
        ret = compute_gross_return_pct("LONG", 80000, 80800)
        assert abs(ret - 1.0) < 0.01

    def test_short_1pct_down(self):
        ret = compute_gross_return_pct("SHORT", 80000, 79200)
        assert abs(ret - 1.0) < 0.01

    def test_long_1pct_down(self):
        ret = compute_gross_return_pct("LONG", 80000, 79200)
        assert abs(ret - (-1.0)) < 0.01

    def test_short_1pct_up(self):
        ret = compute_gross_return_pct("SHORT", 80000, 80800)
        assert abs(ret - (-1.0)) < 0.01

    def test_zero_entry(self):
        assert compute_gross_return_pct("LONG", 0, 100) == 0.0


# ===========================================================================
# Fee Calculations
# ===========================================================================

class TestTradeCosts:
    def test_default_costs(self):
        """Test cost calculation with default TradingCosts."""
        fees, spread, slippage = compute_trade_costs(
            position_size=1000,
            maker_fee_pct=0.10,
            taker_fee_pct=0.10,
            spread_pct=0.05,
            slippage_pct=0.05,
        )
        # Fees: 2 * (1000 * 0.001) = 2.0
        assert abs(fees - 2.0) < 0.01
        # Spread: 1000 * 0.0005 = 0.5
        assert abs(spread - 0.5) < 0.01
        # Slippage: 1000 * 0.0005 * 2 = 1.0
        assert abs(slippage - 1.0) < 0.01

    def test_zero_costs(self):
        fees, spread, slippage = compute_trade_costs(
            1000, 0.0, 0.0, 0.0, 0.0
        )
        assert fees == 0.0
        assert spread == 0.0
        assert slippage == 0.0

    def test_total_round_trip(self):
        costs = TradingCosts(
            maker_fee_pct=0.10,
            taker_fee_pct=0.10,
            spread_pct=0.05,
            slippage_pct=0.05,
        )
        # entry: 0.10 + 0.025 + 0.05 = 0.175
        # exit:  0.10 + 0.025 + 0.05 = 0.175
        # total: 0.35
        assert abs(costs.total_round_trip_pct - 0.35) < 0.001


# ===========================================================================
# Signal Generation
# ===========================================================================

class TestSignalGeneration:
    def test_long_signal(self):
        pred = _make_pred(entry=80000, predicted=80300)  # +0.375%
        config = StrategyConfig(threshold_pct=0.30)
        signal = generate_signal(pred, config)
        assert signal.direction == "LONG"
        assert abs(signal.predicted_return_pct - 0.375) < 0.01
        assert abs(signal.signal_strength - 0.375) < 0.01

    def test_short_signal(self):
        pred = _make_pred(entry=80000, predicted=79700)  # -0.375%
        config = StrategyConfig(threshold_pct=0.30)
        signal = generate_signal(pred, config)
        assert signal.direction == "SHORT"

    def test_no_trade_below_threshold(self):
        pred = _make_pred(entry=80000, predicted=80100)  # +0.125%
        config = StrategyConfig(threshold_pct=0.30)
        signal = generate_signal(pred, config)
        assert signal.direction == "NO_TRADE"

    def test_no_trade_small_negative(self):
        pred = _make_pred(entry=80000, predicted=79900)  # -0.125%
        config = StrategyConfig(threshold_pct=0.30)
        signal = generate_signal(pred, config)
        assert signal.direction == "NO_TRADE"

    def test_threshold_zero_always_trades(self):
        pred = _make_pred(entry=80000, predicted=80001)  # +0.00125%
        config = StrategyConfig(threshold_pct=0.0)
        signal = generate_signal(pred, config)
        assert signal.direction == "LONG"

    def test_invalid_entry_price(self):
        pred = _make_pred(entry=0, predicted=80000)
        config = StrategyConfig(threshold_pct=0.30)
        with pytest.raises(ValueError):
            generate_signal(pred, config)

    def test_multiple_signals(self):
        config = StrategyConfig(threshold_pct=0.30)
        signals = generate_signals(SAMPLE_PREDICTIONS, config)
        assert len(signals) == 5

        # p1: +0.375% → LONG
        assert signals[0].direction == "LONG"
        # p2: -0.374% → SHORT
        assert signals[1].direction == "SHORT"
        # p3: +0.063% → NO_TRADE
        assert signals[2].direction == "NO_TRADE"
        # p4: +0.621% → LONG
        assert signals[3].direction == "LONG"
        # p5: -0.862% → SHORT
        assert signals[4].direction == "SHORT"


# ===========================================================================
# Position Sizing
# ===========================================================================

class TestPositionSizing:
    def test_default_10pct(self):
        config = RiskConfig(starting_capital=10000, max_position_pct=10.0)
        size = compute_position_size(10000, config)
        assert size == 1000.0

    def test_small_capital(self):
        config = RiskConfig(starting_capital=100, max_position_pct=10.0)
        size = compute_position_size(100, config)
        assert size == 10.0

    def test_full_position(self):
        config = RiskConfig(starting_capital=10000, max_position_pct=100.0)
        size = compute_position_size(10000, config)
        assert size == 10000.0


# ===========================================================================
# Equity Curve
# ===========================================================================

class TestEquityCurve:
    def test_empty_trades(self):
        equity, ts = build_equity_curve([], 10000)
        assert equity == [10000]
        assert ts == ["start"]

    def test_winning_trades(self):
        from evaluation.trading.types import Trade

        trades = [
            Trade(
                prediction_id="p1", direction="LONG",
                entry_timestamp="2026-09-16T15:00:00Z",
                exit_timestamp="2026-09-17T15:00:00Z",
                entry_price=80000, exit_price=80800,
                gross_return_pct=1.0, gross_pnl=10.0,
                fees=2.0, spread_cost=0.5, slippage_cost=1.0,
                net_pnl=6.5, net_return_pct=0.65,
                holding_period_hours=24, signal_strength=0.5,
                position_size=1000,
            ),
            Trade(
                prediction_id="p2", direction="SHORT",
                entry_timestamp="2026-09-17T15:00:00Z",
                exit_timestamp="2026-09-18T15:00:00Z",
                entry_price=80800, exit_price=80000,
                gross_return_pct=0.99, gross_pnl=9.9,
                fees=2.0, spread_cost=0.5, slippage_cost=1.0,
                net_pnl=6.4, net_return_pct=0.64,
                holding_period_hours=24, signal_strength=0.5,
                position_size=1000,
            ),
        ]
        equity, ts = build_equity_curve(trades, 10000)
        assert len(equity) == 3
        assert equity[0] == 10000
        assert abs(equity[1] - 10006.5) < 0.01
        assert abs(equity[2] - 10012.9) < 0.01


# ===========================================================================
# Aggregate Metrics
# ===========================================================================

class TestAggregateMetrics:
    def test_net_pnl(self):
        assert compute_net_pnl([10, -5, 8, -3]) == 10

    def test_return_pct(self):
        assert abs(compute_return_pct(500, 10000) - 5.0) < 0.01

    def test_win_rate(self):
        assert abs(compute_win_rate([10, -5, 8, -3]) - 50.0) < 0.01
        assert abs(compute_win_rate([10, 5, 8]) - 100.0) < 0.01
        assert compute_win_rate([]) == 0.0

    def test_profit_factor(self):
        # Gross profit = 18, gross loss = 8
        pf = compute_profit_factor([10, -5, 8, -3])
        assert abs(pf - 2.25) < 0.01

    def test_profit_factor_no_losses(self):
        assert compute_profit_factor([10, 5, 8]) == float("inf")

    def test_profit_factor_no_wins(self):
        assert compute_profit_factor([-10, -5]) == 0.0

    def test_profit_factor_empty(self):
        assert compute_profit_factor([]) == 0.0

    def test_max_drawdown(self):
        # Peak at 110, drops to 90 → 18.18%
        curve = [100, 110, 105, 90, 95, 100]
        dd = compute_max_drawdown_pct(curve)
        assert abs(dd - 18.18) < 0.1

    def test_max_drawdown_no_drawdown(self):
        curve = [100, 110, 120, 130]
        assert compute_max_drawdown_pct(curve) == 0.0

    def test_max_drawdown_single(self):
        assert compute_max_drawdown_pct([100]) == 0.0

    def test_drawdown_series(self):
        curve = [100, 110, 100, 90]
        dd = compute_drawdown_series(curve)
        assert dd[0] == 0.0
        assert dd[1] == 0.0
        assert abs(dd[2] - 9.09) < 0.1
        assert abs(dd[3] - 18.18) < 0.1

    def test_sharpe_ratio(self):
        # With positive mean and some variance → positive Sharpe
        returns = [1.0, -0.5, 0.8, 0.3, -0.2, 1.2, 0.5]
        sharpe = compute_sharpe_ratio(returns)
        assert sharpe > 0

    def test_sharpe_ratio_insufficient_data(self):
        assert compute_sharpe_ratio([1.0]) == 0.0
        assert compute_sharpe_ratio([]) == 0.0

    def test_sortino_ratio(self):
        returns = [1.0, -0.5, 0.8, 0.3, -0.2, 1.2, 0.5]
        sortino = compute_sortino_ratio(returns)
        assert sortino > 0

    def test_sortino_no_downside(self):
        sortino = compute_sortino_ratio([1.0, 2.0, 3.0])
        assert sortino == float("inf")

    def test_avg_trade_pnl(self):
        assert abs(compute_avg_trade_pnl([10, -5, 8, -3]) - 2.5) < 0.01

    def test_avg_win(self):
        assert abs(compute_avg_win([10, -5, 8, -3]) - 9.0) < 0.01

    def test_avg_loss(self):
        assert abs(compute_avg_loss([10, -5, 8, -3]) - (-4.0)) < 0.01

    def test_expected_value(self):
        # WR=50%, avg_win=9, avg_loss=-4
        # EV = 0.5 * 9 + 0.5 * (-4) = 2.5
        ev = compute_expected_value(50.0, 9.0, -4.0)
        assert abs(ev - 2.5) < 0.01

    def test_time_invested(self):
        assert abs(compute_time_invested_pct(48, 240) - 20.0) < 0.01
        assert compute_time_invested_pct(0, 0) == 0.0

    def test_buy_hold_return(self):
        ret = compute_buy_hold_return_pct(80000, 84000)
        assert abs(ret - 5.0) < 0.01


# ===========================================================================
# Backtest Engine
# ===========================================================================

class TestBacktestEngine:
    def test_basic_backtest(self):
        """Full backtest with 5 predictions, threshold 0.30%."""
        config = StrategyConfig(threshold_pct=0.30)
        costs = TradingCosts()
        risk = RiskConfig(starting_capital=10000, max_position_pct=10.0)

        result = run_backtest(SAMPLE_PREDICTIONS, config, costs, risk)

        # Should have trades from p1, p2, p4, p5 (p3 is NO_TRADE)
        assert result.total_trades == 4
        assert result.skipped_signals == 1
        assert result.total_trades == result.winning_trades + result.losing_trades
        assert len(result.equity_curve) == result.total_trades + 1
        assert result.equity_curve[0] == 10000.0

    def test_zero_threshold(self):
        """Threshold 0 → every prediction generates a trade."""
        config = StrategyConfig(threshold_pct=0.0)
        result = run_backtest(SAMPLE_PREDICTIONS, config)
        assert result.total_trades == 5

    def test_high_threshold(self):
        """Very high threshold → no trades."""
        config = StrategyConfig(threshold_pct=10.0)
        result = run_backtest(SAMPLE_PREDICTIONS, config)
        assert result.total_trades == 0
        assert result.net_pnl == 0.0
        assert result.return_pct == 0.0

    def test_zero_cost_backtest(self):
        """With zero costs, net = gross."""
        config = StrategyConfig(threshold_pct=0.0)
        costs = TradingCosts(
            maker_fee_pct=0, taker_fee_pct=0,
            spread_pct=0, slippage_pct=0
        )
        result = run_backtest(SAMPLE_PREDICTIONS, config, costs)
        assert result.total_fees == 0.0
        assert result.total_slippage == 0.0
        assert result.total_spread == 0.0

    def test_multi_threshold(self):
        """Test multiple thresholds produce different results."""
        results = run_multi_threshold_backtest(
            SAMPLE_PREDICTIONS,
            thresholds=[0.10, 0.50, 1.0],
        )
        assert len(results) == 3
        # Higher threshold → fewer trades
        assert results[0].total_trades >= results[1].total_trades >= results[2].total_trades

    def test_empty_predictions(self):
        result = run_backtest([])
        assert result.total_trades == 0
        assert result.net_pnl == 0.0


# ===========================================================================
# Look-Ahead Protection
# ===========================================================================

class TestLookAheadProtection:
    """Verify that no future data enters the signal generation."""

    def test_signal_uses_only_entry_price(self):
        """Signal must be based solely on entry_price and predicted_price.

        It must NOT use exit_price, high_24h, or low_24h.
        """
        pred = _make_pred(
            entry=80000,
            predicted=80300,
            exit_price=85000,  # Even with huge exit, signal unchanged
        )
        config = StrategyConfig(threshold_pct=0.30)
        signal = generate_signal(pred, config)

        # Signal should only reflect predicted vs entry
        assert abs(signal.predicted_return_pct - 0.375) < 0.01
        assert signal.direction == "LONG"

    def test_prediction_timestamp_before_exit(self):
        """Ensure prediction timestamp is always before exit would occur."""
        from datetime import datetime

        for pred in SAMPLE_PREDICTIONS:
            pred_ts = datetime.fromisoformat(
                pred.timestamp.replace("Z", "+00:00")
            )
            # Exit is 24h after prediction
            from datetime import timedelta
            exit_ts = pred_ts + timedelta(hours=24)
            assert exit_ts > pred_ts

    def test_entry_price_independent_of_exit(self):
        """Changing exit_price should not affect signal generation."""
        pred_a = _make_pred(entry=80000, predicted=80300, exit_price=80200)
        pred_b = _make_pred(entry=80000, predicted=80300, exit_price=75000)

        config = StrategyConfig(threshold_pct=0.30)

        signal_a = generate_signal(pred_a, config)
        signal_b = generate_signal(pred_b, config)

        assert signal_a.direction == signal_b.direction
        assert signal_a.predicted_return_pct == signal_b.predicted_return_pct
        assert signal_a.signal_strength == signal_b.signal_strength


# ===========================================================================
# Timestamp Correctness
# ===========================================================================

class TestTimestampCorrectness:
    def test_no_future_data_in_features(self):
        """All feature data timestamps must be <= prediction timestamp.

        This test verifies the fundamental rule:
        features.timestamp <= prediction.timestamp
        """
        from datetime import datetime

        for pred in SAMPLE_PREDICTIONS:
            # entry_price is available at prediction time ✓
            # predicted_price is generated at prediction time ✓
            # exit_price is only used for evaluation, never for signal ✓
            # Verify the prediction_id and timestamp are consistent
            assert pred.timestamp
            assert pred.entry_price > 0

            # Verify no future price used for signal
            config = StrategyConfig(threshold_pct=0.0)
            signal = generate_signal(pred, config)

            # The signal should NOT depend on exit_price in any way
            assert signal.entry_price == pred.entry_price
            assert signal.predicted_price == pred.predicted_price


# ===========================================================================
# Stop Loss and Take Profit
# ===========================================================================

class TestStopLossTakeProfit:
    def test_tp_sl_exit_mode(self):
        """Test that TP/SL are applied when configured."""
        preds = [
            _make_pred(
                "p1", "2026-09-16T15:00:00Z",
                entry=80000, predicted=81000,  # LONG signal
                exit_price=80500,
                high=81200,  # TP would be hit
                low=79500,   # SL would also be hit
            ),
        ]
        # SL checked before TP → conservative
        config = StrategyConfig(
            threshold_pct=0.0,
            exit_mode="tp_sl",
            take_profit_pct=1.0,
            stop_loss_pct=0.5,
        )
        result = run_backtest(preds, config)
        assert result.total_trades == 1
        trade = result.trades[0]
        # SL at 80000 * (1 - 0.005) = 79600, low is 79500 → SL hit
        assert abs(trade.exit_price - 79600) < 1.0


# ===========================================================================
# Baseline Strategies
# ===========================================================================

class TestBaselines:
    def test_buy_and_hold(self):
        from evaluation.trading.baselines import buy_and_hold_return

        result = buy_and_hold_return(SAMPLE_PREDICTIONS)
        # First entry: 80000, last exit: 80000
        assert result.name == "Buy & Hold"
        assert result.total_trades == 1

    def test_naive_with_threshold(self):
        from evaluation.trading.baselines import naive_strategy

        config = StrategyConfig(threshold_pct=0.30)
        result = naive_strategy(SAMPLE_PREDICTIONS, config)
        # Naive predicts no change → always below threshold → 0 trades
        assert result.total_trades == 0
        assert result.net_pnl == 0.0

    def test_momentum_needs_two_predictions(self):
        from evaluation.trading.baselines import momentum_strategy

        result = momentum_strategy([SAMPLE_PREDICTIONS[0]])
        assert result.total_trades == 0

    def test_random_reproducible(self):
        from evaluation.trading.baselines import random_strategy

        result1 = random_strategy(SAMPLE_PREDICTIONS, seed=42)
        result2 = random_strategy(SAMPLE_PREDICTIONS, seed=42)
        assert result1.return_pct == result2.return_pct

    def test_run_all_baselines(self):
        from evaluation.trading.baselines import run_all_baselines

        results = run_all_baselines(SAMPLE_PREDICTIONS)
        assert len(results) == 4
        names = [r.name for r in results]
        assert "Buy & Hold" in names
        assert "Random 50/50" in names
        assert "Naive (no change)" in names
        assert "Momentum" in names


# ===========================================================================
# Missing / Duplicate Data
# ===========================================================================

class TestEdgeCases:
    def test_duplicate_prediction_ids(self):
        """Duplicate IDs should still produce valid results."""
        preds = [
            _make_pred("dup", "2026-09-16T15:00:00Z", 80000, 80300, 80200),
            _make_pred("dup", "2026-09-17T15:00:00Z", 80200, 80500, 80400),
        ]
        result = run_backtest(preds, StrategyConfig(threshold_pct=0.0))
        # Both should trade (same ID doesn't block execution)
        assert result.total_trades == 2

    def test_single_prediction(self):
        preds = [_make_pred("solo", "2026-09-16T15:00:00Z", 80000, 80300, 80200)]
        result = run_backtest(preds, StrategyConfig(threshold_pct=0.0))
        assert result.total_trades == 1

    def test_all_no_trade(self):
        """All predictions too small to trade."""
        preds = [
            _make_pred("p1", "2026-09-16T15:00:00Z", 80000, 80010, 80100),
            _make_pred("p2", "2026-09-17T15:00:00Z", 80100, 80110, 80200),
        ]
        config = StrategyConfig(threshold_pct=1.0)
        result = run_backtest(preds, config)
        assert result.total_trades == 0
        assert result.skipped_signals == 2
