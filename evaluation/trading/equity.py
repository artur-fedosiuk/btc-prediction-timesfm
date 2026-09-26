"""
Equity curve computation.

Tracks portfolio value over time as trades are executed.
Handles position sizing based on RiskConfig.
"""

from __future__ import annotations

from .types import RiskConfig, Trade


def compute_position_size(
    current_capital: float,
    config: RiskConfig,
) -> float:
    """Compute position size for a trade based on risk config.

    Args:
        current_capital: Current portfolio value.
        config: Risk configuration with max position %.

    Returns:
        Dollar amount to allocate to the trade.
    """
    return current_capital * config.max_position_pct / 100.0


def build_equity_curve(
    trades: list[Trade],
    starting_capital: float,
) -> tuple[list[float], list[str]]:
    """Build equity curve from a sequence of trades.

    Each point in the curve represents the portfolio value
    after a trade is closed.

    Args:
        trades: Chronologically sorted trades.
        starting_capital: Initial portfolio value.

    Returns:
        Tuple of (equity_values, timestamps) including the starting point.
    """
    equity = [starting_capital]
    timestamps = ["start"]
    current = starting_capital

    for trade in trades:
        current += trade.net_pnl
        equity.append(current)
        timestamps.append(trade.exit_timestamp)

    return equity, timestamps
