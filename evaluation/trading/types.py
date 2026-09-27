"""
Core data types for the trading strategy evaluation system.

All types are immutable dataclasses used across the backtesting pipeline.
These types enforce a clean separation between prediction quality
and trading profitability measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Configuration types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradingCosts:
    """Real-world trading cost model.

    All values are percentages (e.g. 0.10 = 0.10% = 10 bps).
    Applied to both entry and exit of every trade.
    """

    maker_fee_pct: float = 0.10
    taker_fee_pct: float = 0.10
    spread_pct: float = 0.05
    slippage_pct: float = 0.05

    @property
    def total_round_trip_pct(self) -> float:
        """Total cost for one complete trade (entry + exit)."""
        entry_cost = self.taker_fee_pct + self.spread_pct / 2 + self.slippage_pct
        exit_cost = self.taker_fee_pct + self.spread_pct / 2 + self.slippage_pct
        return entry_cost + exit_cost


@dataclass(frozen=True)
class StrategyConfig:
    """Trading strategy parameters.

    threshold_pct: Minimum predicted return to trigger a trade.
    position_mode: How overlapping signals are handled.
        - 'single': One position at a time, ignore new signals while open.
        - 'independent': Each prediction is an independent trade.
    exit_mode: How trades are exited.
        - 'time': Exit after holding_period_hours.
        - 'tp_sl': Exit at take profit or stop loss.
    """

    accounting_mode: Literal[
        "INDEPENDENT_HYPOTHETICAL_TRADES", "INVESTABLE_PORTFOLIO"
    ] = "INDEPENDENT_HYPOTHETICAL_TRADES"
    threshold_pct: float = 0.30
    position_mode: Literal["single", "independent"] = "independent"
    exit_mode: Literal["time", "tp_sl"] = "time"
    holding_period_hours: float = 24.0
    take_profit_pct: float | None = None
    stop_loss_pct: float | None = None
    version: str = "v1"
    description: str = ""


@dataclass(frozen=True)
class RiskConfig:
    """Position sizing and risk management parameters."""

    starting_capital: float = 10_000.0
    risk_per_trade_pct: float = 2.0
    max_position_pct: float = 10.0


# ---------------------------------------------------------------------------
# Signal type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradingSignal:
    """Output of the signal generator for a single prediction."""

    prediction_id: str
    timestamp: str
    direction: Literal["LONG", "SHORT", "NO_TRADE"]
    predicted_return_pct: float
    entry_price: float
    predicted_price: float
    signal_strength: float  # abs(predicted_return_pct)


# ---------------------------------------------------------------------------
# Trade type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trade:
    """A single completed trade with full cost breakdown.

    Every trade is immutable once created — results are never modified
    after the fact.
    """

    prediction_id: str
    direction: Literal["LONG", "SHORT"]
    entry_timestamp: str
    exit_timestamp: str
    entry_price: float
    exit_price: float

    # Gross performance
    gross_return_pct: float
    gross_pnl: float

    # Cost breakdown
    fees: float
    spread_cost: float
    slippage_cost: float

    # Net performance
    net_pnl: float
    net_return_pct: float

    # Metadata
    holding_period_hours: float
    signal_strength: float
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    exit_notional: float = 0.0
    position_size: float = 0.0  # Dollar amount of the position


# ---------------------------------------------------------------------------
# Backtest result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestResult:
    """Complete results of a backtest run.

    Contains all trades, equity curve, and aggregate metrics.
    The strategy_config, trading_costs, and risk_config that
    produced this result are included for reproducibility.
    """

    # Trades
    trades: tuple[Trade, ...]
    equity_curve: tuple[float, ...]
    equity_timestamps: tuple[str, ...]

    # Aggregate metrics
    net_pnl: float
    return_pct: float
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    avg_trade_pnl: float
    expected_value: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    time_invested_pct: float
    buy_hold_return_pct: float

    # Cost totals
    total_fees: float
    total_slippage: float
    total_spread: float

    # Signals that were skipped (NO_TRADE)
    skipped_signals: int

    # Configuration (for reproducibility)
    strategy_config: StrategyConfig
    trading_costs: TradingCosts
    risk_config: RiskConfig


# ---------------------------------------------------------------------------
# Prediction data for backtesting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionRecord:
    """A single prediction with all required market data for backtesting.

    Contains only data available at or before prediction time (entry_price)
    and market outcome data that is only used for evaluation (exit_price,
    high_24h, low_24h).
    """

    prediction_id: str
    timestamp: str
    entry_price: float
    predicted_price: float
    predicted_min: float
    predicted_max: float
    exit_price: float  # Price at T + holding_period
    high_24h: float | None = None
    low_24h: float | None = None
    twap_24h: float | None = None
    prediction_method: str = ""
