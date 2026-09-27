"""
Backtesting engine for the trading strategy evaluation.

Walks through predictions chronologically, generates signals,
simulates trades with full cost model, and produces a complete
BacktestResult with all metrics and equity curve.

Paper accounting only; historical entry/exit observations are assumed executable.
Independent diagnostics are distinct from a capital-constrained portfolio.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta

from evaluation.forward.contracts import utc

from .equity import build_equity_curve, compute_position_size
from .metrics import (
    compute_avg_loss,
    compute_avg_trade_pnl,
    compute_avg_win,
    compute_buy_hold_return_pct,
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
from .signal import generate_signals
from .types import (
    BacktestResult,
    PredictionRecord,
    RiskConfig,
    StrategyConfig,
    Trade,
    TradingCosts,
)

logger = logging.getLogger(__name__)


def _parse_timestamp(ts: str) -> datetime:
    """Parse an ISO timestamp string, tolerating 'Z' suffix."""
    return utc(ts)


def _compute_exit_timestamp(entry_ts: str, hours: float) -> str:
    """Compute exit timestamp by adding holding period to entry."""
    dt = _parse_timestamp(entry_ts)
    exit_dt = dt + timedelta(hours=hours)
    return exit_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def run_backtest(
    predictions: list[PredictionRecord],
    strategy_config: StrategyConfig | None = None,
    trading_costs: TradingCosts | None = None,
    risk_config: RiskConfig | None = None,
) -> BacktestResult:
    """Run a complete backtest on a list of predictions.

    This is the main entry point for backtesting. It:
    1. Generates signals from predictions
    2. Filters by threshold
    3. Simulates each trade with position sizing
    4. Computes full cost model
    5. Builds equity curve
    6. Computes all aggregate metrics

    Args:
        predictions: Chronologically sorted prediction records.
            Each must have entry_price and exit_price filled in.
        strategy_config: Strategy parameters (threshold, position mode).
        trading_costs: Cost model.
        risk_config: Position sizing and risk parameters.

    Returns:
        BacktestResult with all metrics, trades, and equity curve.
    """
    strategy_config = strategy_config or StrategyConfig()
    trading_costs = trading_costs or TradingCosts()
    risk_config = risk_config or RiskConfig()

    predictions = sorted(predictions, key=lambda p: utc(p.timestamp))
    if not 0 < risk_config.max_position_pct <= 100 or risk_config.starting_capital <= 0:
        raise ValueError("Invalid risk configuration")
    if strategy_config.holding_period_hours <= 0:
        raise ValueError("Invalid holding period")
    for pred in predictions:
        if not math.isfinite(pred.exit_price) or pred.exit_price <= 0:
            raise ValueError("Invalid exit price")
    for cost in (
        trading_costs.taker_fee_pct,
        trading_costs.spread_pct,
        trading_costs.slippage_pct,
    ):
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("Invalid trading cost")
    portfolio = strategy_config.accounting_mode == "INVESTABLE_PORTFOLIO"
    if portfolio and strategy_config.exit_mode != "time":
        raise ValueError("TP/SL timing is unknown; hypothetical mode only")
    # 1. Generate signals
    signals = generate_signals(predictions, strategy_config)

    # 2. Build prediction lookup for exit prices
    pred_by_id = {p.prediction_id: p for p in predictions}

    # 3. Execute trades
    trades: list[Trade] = []
    current_capital = risk_config.starting_capital
    skipped = 0
    open_trades = []
    entry_cost_rate = (
        trading_costs.taker_fee_pct
        + trading_costs.spread_pct / 2
        + trading_costs.slippage_pct
    ) / 100

    for signal in signals:
        if signal.direction == "NO_TRADE":
            skipped += 1
            continue

        pred = pred_by_id.get(signal.prediction_id)
        if pred is None:
            logger.warning("No prediction found for signal %s", signal.prediction_id)
            skipped += 1
            continue

        # Realize only exits that occurred before this entry. Reserve open notional.
        if portfolio:
            matured = [
                t for t in open_trades if utc(t.exit_timestamp) <= utc(signal.timestamp)
            ]
            current_capital += sum(
                t.net_pnl + t.position_size * entry_cost_rate for t in matured
            )
            open_trades = [t for t in open_trades if t not in matured]
        sizing_capital = current_capital if portfolio else risk_config.starting_capital
        position_size = compute_position_size(sizing_capital, risk_config)
        if portfolio:
            reserve_rate = (
                trading_costs.taker_fee_pct
                + trading_costs.spread_pct / 2
                + trading_costs.slippage_pct
            ) / 100
            available = current_capital - sum(t.position_size for t in open_trades)
            position_size = min(position_size, max(0, available) / (1 + reserve_rate))
        if position_size <= 0:
            logger.warning("Position size is 0 — capital depleted")
            skipped += 1
            continue

        entry_price = signal.entry_price
        exit_price = pred.exit_price

        # Check for TP/SL exit if configured
        if strategy_config.exit_mode == "tp_sl" and pred.high_24h and pred.low_24h:
            exit_price = _apply_tp_sl(
                direction=signal.direction,
                entry_price=entry_price,
                high_24h=pred.high_24h,
                low_24h=pred.low_24h,
                exit_price_default=pred.exit_price,
                take_profit_pct=strategy_config.take_profit_pct,
                stop_loss_pct=strategy_config.stop_loss_pct,
            )

        # Gross PnL
        gross_return_pct = compute_gross_return_pct(
            signal.direction, entry_price, exit_price
        )
        gross_pnl = compute_gross_pnl(
            signal.direction, entry_price, exit_price, position_size
        )

        # Costs
        fees, spread_cost, slippage_cost = compute_trade_costs(
            position_size,
            trading_costs.maker_fee_pct,
            trading_costs.taker_fee_pct,
            trading_costs.spread_pct,
            trading_costs.slippage_pct,
            exit_notional=position_size * exit_price / entry_price,
        )
        total_costs = fees + spread_cost + slippage_cost

        net_pnl = gross_pnl - total_costs
        net_return_pct = (net_pnl / position_size * 100.0) if position_size > 0 else 0.0

        exit_timestamp = _compute_exit_timestamp(
            signal.timestamp, strategy_config.holding_period_hours
        )

        trade = Trade(
            prediction_id=signal.prediction_id,
            direction=signal.direction,
            entry_timestamp=signal.timestamp,
            exit_timestamp=exit_timestamp,
            entry_price=entry_price,
            exit_price=exit_price,
            gross_return_pct=gross_return_pct,
            gross_pnl=gross_pnl,
            fees=fees,
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            net_pnl=net_pnl,
            net_return_pct=net_return_pct,
            holding_period_hours=strategy_config.holding_period_hours,
            signal_strength=signal.signal_strength,
            position_size=position_size,
            entry_fee=position_size * trading_costs.taker_fee_pct / 100,
            exit_fee=position_size
            * exit_price
            / entry_price
            * trading_costs.taker_fee_pct
            / 100,
            exit_notional=position_size * exit_price / entry_price,
        )

        trades.append(trade)
        open_trades.append(trade)
        if portfolio:
            current_capital -= position_size * entry_cost_rate

    # 4. Build equity curve
    equity_values, equity_timestamps = build_equity_curve(
        sorted(trades, key=lambda t: utc(t.exit_timestamp)),
        risk_config.starting_capital,
    )

    # 5. Compute aggregate metrics
    trade_pnls = [t.net_pnl for t in trades]
    trade_returns = [t.net_return_pct for t in trades]
    total_trades = len(trades)

    net_pnl_total = compute_net_pnl(trade_pnls)
    return_pct = compute_return_pct(net_pnl_total, risk_config.starting_capital)
    win_rate = compute_win_rate(trade_pnls)
    profit_factor = compute_profit_factor(trade_pnls)
    max_dd = compute_max_drawdown_pct(equity_values)
    sharpe = compute_sharpe_ratio(trade_returns)
    sortino = compute_sortino_ratio(trade_returns)
    avg_pnl = compute_avg_trade_pnl(trade_pnls)
    avg_win = compute_avg_win(trade_pnls)
    avg_loss = compute_avg_loss(trade_pnls)
    ev = compute_expected_value(win_rate, avg_win, avg_loss)

    winning = sum(1 for p in trade_pnls if p > 0)
    losing = sum(1 for p in trade_pnls if p < 0)

    # Time invested
    total_holding = sum(t.holding_period_hours for t in trades)
    if predictions:
        try:
            first_ts = _parse_timestamp(predictions[0].timestamp)
            last_ts = _parse_timestamp(predictions[-1].timestamp)
            period_hours = max(
                (last_ts - first_ts).total_seconds() / 3600.0,
                1.0,
            )
        except ValueError:
            period_hours = max(total_holding, 1.0)
    else:
        period_hours = max(total_holding, 1.0)
    time_invested = compute_time_invested_pct(total_holding, period_hours)

    # Buy & Hold baseline
    if predictions:
        bh_return = compute_buy_hold_return_pct(
            predictions[0].entry_price,
            predictions[-1].exit_price,
        )
    else:
        bh_return = 0.0

    total_fees = sum(t.fees for t in trades)
    total_slippage = sum(t.slippage_cost for t in trades)
    total_spread = sum(t.spread_cost for t in trades)

    return BacktestResult(
        trades=tuple(trades),
        equity_curve=tuple(equity_values),
        equity_timestamps=tuple(equity_timestamps),
        net_pnl=net_pnl_total,
        return_pct=return_pct,
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown_pct=max_dd,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        avg_trade_pnl=avg_pnl,
        expected_value=ev,
        total_trades=total_trades,
        winning_trades=winning,
        losing_trades=losing,
        time_invested_pct=time_invested,
        buy_hold_return_pct=bh_return,
        total_fees=total_fees,
        total_slippage=total_slippage,
        total_spread=total_spread,
        skipped_signals=skipped,
        strategy_config=strategy_config,
        trading_costs=trading_costs,
        risk_config=risk_config,
    )


def _apply_tp_sl(
    direction: str,
    entry_price: float,
    high_24h: float,
    low_24h: float,
    exit_price_default: float,
    take_profit_pct: float | None,
    stop_loss_pct: float | None,
) -> float:
    """Apply take-profit and stop-loss exit logic.

    Uses high/low of the 24h period to determine if TP or SL
    would have been hit. Conservative assumption: SL is checked
    before TP (worst case for the trader).

    Returns:
        The exit price after TP/SL logic.
    """
    if direction == "LONG":
        # Stop loss hit?
        if stop_loss_pct is not None:
            sl_price = entry_price * (1 - stop_loss_pct / 100.0)
            if low_24h <= sl_price:
                return sl_price
        # Take profit hit?
        if take_profit_pct is not None:
            tp_price = entry_price * (1 + take_profit_pct / 100.0)
            if high_24h >= tp_price:
                return tp_price
    elif direction == "SHORT":
        # Stop loss hit? (price goes up)
        if stop_loss_pct is not None:
            sl_price = entry_price * (1 + stop_loss_pct / 100.0)
            if high_24h >= sl_price:
                return sl_price
        # Take profit hit? (price goes down)
        if take_profit_pct is not None:
            tp_price = entry_price * (1 - take_profit_pct / 100.0)
            if low_24h <= tp_price:
                return tp_price

    return exit_price_default


def run_multi_threshold_backtest(
    predictions: list[PredictionRecord],
    thresholds: list[float] | None = None,
    trading_costs: TradingCosts | None = None,
    risk_config: RiskConfig | None = None,
    base_config: StrategyConfig | None = None,
) -> list[BacktestResult]:
    """Run backtests across multiple threshold values.

    Args:
        predictions: Sorted predictions.
        thresholds: List of threshold percentages to test.
        trading_costs: Cost model.
        risk_config: Risk config.
        base_config: Base strategy config (threshold will be overridden).

    Returns:
        List of BacktestResults, one per threshold.
    """
    if thresholds is None:
        thresholds = [0.10, 0.20, 0.30, 0.50, 0.75, 1.00]

    base = base_config or StrategyConfig()
    results = []

    for thresh in thresholds:
        config = StrategyConfig(
            threshold_pct=thresh,
            position_mode=base.position_mode,
            exit_mode=base.exit_mode,
            holding_period_hours=base.holding_period_hours,
            take_profit_pct=base.take_profit_pct,
            stop_loss_pct=base.stop_loss_pct,
            version=f"{base.version}_t{thresh}",
            description=f"Threshold {thresh}%",
        )
        result = run_backtest(predictions, config, trading_costs, risk_config)
        results.append(result)
        logger.info(
            "Threshold %.2f%%: %d trades, PnL=$%.2f (%.2f%%), WR=%.1f%%, PF=%.2f, DD=%.1f%%",
            thresh,
            result.total_trades,
            result.net_pnl,
            result.return_pct,
            result.win_rate,
            result.profit_factor,
            result.max_drawdown_pct,
        )

    return results
