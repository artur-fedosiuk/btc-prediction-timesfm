"""
Trading performance metrics.

All functions are stateless and deterministic — easy to unit-test.
Computes trading-specific metrics: PnL, Sharpe, Sortino, drawdown,
profit factor, win rate, etc.

These metrics answer: "Would this strategy make money?"
as opposed to prediction metrics which answer: "Was the prediction accurate?"
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Trade-level PnL
# ---------------------------------------------------------------------------


def compute_gross_pnl(
    direction: str,
    entry_price: float,
    exit_price: float,
    position_size: float,
) -> float:
    """Compute gross PnL for a single trade.

    Args:
        direction: 'LONG' or 'SHORT'.
        entry_price: Price at entry.
        exit_price: Price at exit.
        position_size: Dollar amount of the position.

    Returns:
        Gross PnL in dollars.
    """
    if direction == "LONG":
        return position_size * (exit_price - entry_price) / entry_price
    elif direction == "SHORT":
        return position_size * (entry_price - exit_price) / entry_price
    else:
        raise ValueError(f"Invalid direction: {direction}")


def compute_gross_return_pct(
    direction: str,
    entry_price: float,
    exit_price: float,
) -> float:
    """Compute gross return percentage for a trade.

    Args:
        direction: 'LONG' or 'SHORT'.
        entry_price: Price at entry.
        exit_price: Price at exit.

    Returns:
        Gross return as a percentage.
    """
    if entry_price <= 0:
        return 0.0
    if direction == "LONG":
        return (exit_price - entry_price) / entry_price * 100.0
    elif direction == "SHORT":
        return (entry_price - exit_price) / entry_price * 100.0
    else:
        raise ValueError(f"Invalid direction: {direction}")


def compute_trade_costs(
    position_size: float,
    maker_fee_pct: float,
    taker_fee_pct: float,
    spread_pct: float,
    slippage_pct: float,
    exit_notional: float | None = None,
) -> tuple[float, float, float]:
    """Compute cost breakdown for a single trade (entry + exit).

    Returns:
        Tuple of (total_fees, spread_cost, slippage_cost).
    """
    exit_notional = position_size if exit_notional is None else exit_notional
    # Entry: taker fee + half spread + slippage
    entry_fee = position_size * taker_fee_pct / 100.0
    # Exit: taker fee + half spread + slippage
    exit_fee = exit_notional * taker_fee_pct / 100.0
    total_fees = entry_fee + exit_fee

    spread_cost = (position_size + exit_notional) * spread_pct / 200.0
    slippage_cost = (
        (position_size + exit_notional) * slippage_pct / 100.0
    )  # entry + exit

    return total_fees, spread_cost, slippage_cost


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def compute_net_pnl(trade_pnls: Sequence[float]) -> float:
    """Sum of all net PnLs."""
    return sum(trade_pnls)


def compute_return_pct(net_pnl: float, starting_capital: float) -> float:
    """Total return percentage on starting capital."""
    if starting_capital <= 0:
        return 0.0
    return net_pnl / starting_capital * 100.0


def compute_win_rate(trade_pnls: Sequence[float]) -> float:
    """Percentage of trades with positive PnL.

    Returns 0.0 for empty list.
    """
    if not trade_pnls:
        return 0.0
    winners = sum(1 for p in trade_pnls if p > 0)
    return winners / len(trade_pnls) * 100.0


def compute_profit_factor(trade_pnls: Sequence[float]) -> float:
    """Ratio of gross profits to gross losses.

    Returns float('inf') if no losing trades.
    Returns 0.0 if no winning trades.
    Returns 0.0 for empty list.
    """
    if not trade_pnls:
        return 0.0
    gross_profit = sum(p for p in trade_pnls if p > 0)
    gross_loss = abs(sum(p for p in trade_pnls if p < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def compute_max_drawdown_pct(equity_curve: Sequence[float]) -> float:
    """Maximum drawdown as a percentage from peak.

    Args:
        equity_curve: Sequence of equity values over time.

    Returns:
        Maximum drawdown percentage (positive number).
        Returns 0.0 for empty or single-element curves.
    """
    if len(equity_curve) < 2:
        return 0.0

    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            dd = (peak - value) / peak * 100.0
            max_dd = max(max_dd, dd)
    return max_dd


def compute_drawdown_series(equity_curve: Sequence[float]) -> list[float]:
    """Compute drawdown percentage at each point in the equity curve.

    Returns:
        List of drawdown percentages (positive = underwater).
    """
    if not equity_curve:
        return []

    peak = equity_curve[0]
    drawdowns = []
    for value in equity_curve:
        peak = max(peak, value)
        dd = (peak - value) / peak * 100.0 if peak > 0 else 0.0
        drawdowns.append(dd)
    return drawdowns


def compute_sharpe_ratio(
    trade_returns_pct: Sequence[float],
    risk_free_rate_annual: float = 0.0,
    trades_per_year: float = 365.0,
) -> float:
    """Annualized Sharpe ratio from trade return percentages.

    Args:
        trade_returns_pct: Individual trade returns (%).
        risk_free_rate_annual: Annual risk-free rate (%).
        trades_per_year: Assumed number of trades per year for annualization.

    Returns:
        Annualized Sharpe ratio. Returns 0.0 if insufficient data.
    """
    if len(trade_returns_pct) < 2:
        return 0.0

    mean_return = sum(trade_returns_pct) / len(trade_returns_pct)
    rf_per_trade = risk_free_rate_annual / trades_per_year

    variance = sum((r - mean_return) ** 2 for r in trade_returns_pct) / (
        len(trade_returns_pct) - 1
    )
    std_dev = math.sqrt(variance)

    if std_dev == 0:
        return 0.0

    sharpe_per_trade = (mean_return - rf_per_trade) / std_dev
    return sharpe_per_trade * math.sqrt(trades_per_year)


def compute_sortino_ratio(
    trade_returns_pct: Sequence[float],
    risk_free_rate_annual: float = 0.0,
    trades_per_year: float = 365.0,
) -> float:
    """Annualized Sortino ratio — penalizes only downside volatility.

    Args:
        trade_returns_pct: Individual trade returns (%).
        risk_free_rate_annual: Annual risk-free rate (%).
        trades_per_year: Assumed number of trades per year for annualization.

    Returns:
        Annualized Sortino ratio. Returns 0.0 if insufficient data.
    """
    if len(trade_returns_pct) < 2:
        return 0.0

    mean_return = sum(trade_returns_pct) / len(trade_returns_pct)
    rf_per_trade = risk_free_rate_annual / trades_per_year

    # Downside deviation: only negative returns
    downside_returns = [r for r in trade_returns_pct if r < 0]
    if not downside_returns:
        return float("inf") if mean_return > rf_per_trade else 0.0

    downside_variance = sum(r**2 for r in downside_returns) / len(trade_returns_pct)
    downside_dev = math.sqrt(downside_variance)

    if downside_dev == 0:
        return 0.0

    sortino_per_trade = (mean_return - rf_per_trade) / downside_dev
    return sortino_per_trade * math.sqrt(trades_per_year)


def compute_avg_trade_pnl(trade_pnls: Sequence[float]) -> float:
    """Average PnL per trade."""
    if not trade_pnls:
        return 0.0
    return sum(trade_pnls) / len(trade_pnls)


def compute_avg_win(trade_pnls: Sequence[float]) -> float:
    """Average PnL of winning trades."""
    winners = [p for p in trade_pnls if p > 0]
    if not winners:
        return 0.0
    return sum(winners) / len(winners)


def compute_avg_loss(trade_pnls: Sequence[float]) -> float:
    """Average PnL of losing trades (returned as negative)."""
    losers = [p for p in trade_pnls if p < 0]
    if not losers:
        return 0.0
    return sum(losers) / len(losers)


def compute_expected_value(
    win_rate_pct: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """Expected value per trade.

    EV = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    """
    wr = win_rate_pct / 100.0
    return (wr * avg_win) + ((1 - wr) * avg_loss)


def compute_time_invested_pct(
    total_holding_hours: float,
    total_period_hours: float,
) -> float:
    """Percentage of time that capital was deployed.

    Returns 0.0 if total_period_hours is zero.
    """
    if total_period_hours <= 0:
        return 0.0
    return min(total_holding_hours / total_period_hours * 100.0, 100.0)


def compute_buy_hold_return_pct(
    first_price: float,
    last_price: float,
) -> float:
    """Buy & Hold return over the period."""
    if first_price <= 0:
        return 0.0
    return (last_price - first_price) / first_price * 100.0
