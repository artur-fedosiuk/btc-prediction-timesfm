"""
Baseline strategies for comparison.

TimesFM must beat these simple strategies after costs to demonstrate
a real edge. If it can't, the prediction quality is insufficient
for profitable trading.

Baselines:
    A — Buy & Hold: Buy at start, hold until end.
    B — Random: Random LONG/SHORT 50/50 (Monte Carlo).
    C — Naive: Predict price = current price (always NO_TRADE with threshold).
    D — Momentum: If price went up last 24h → LONG, else → SHORT.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from .backtest_engine import run_backtest
from .types import (
    BacktestResult,
    PredictionRecord,
    RiskConfig,
    StrategyConfig,
    TradingCosts,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaselineResult:
    """Result of a baseline strategy."""

    name: str
    description: str
    return_pct: float
    net_pnl: float
    total_trades: int
    win_rate: float
    max_drawdown_pct: float
    sharpe_ratio: float


# ---------------------------------------------------------------------------
# Baseline A — Buy & Hold
# ---------------------------------------------------------------------------

def buy_and_hold_return(
    predictions: list[PredictionRecord],
    starting_capital: float = 10_000.0,
    costs: TradingCosts | None = None,
) -> BaselineResult:
    """Compute Buy & Hold return over the prediction period.

    Buys at the entry_price of the first prediction and sells at
    the exit_price of the last prediction. Applies entry and exit costs.

    Args:
        predictions: Sorted predictions covering the period.
        starting_capital: Initial capital.
        costs: Trading costs (applied once for entry and once for exit).

    Returns:
        BaselineResult for Buy & Hold.
    """
    if not predictions:
        return BaselineResult(
            name="Buy & Hold",
            description="Buy BTC at start, hold until end",
            return_pct=0.0,
            net_pnl=0.0,
            total_trades=1,
            win_rate=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
        )

    costs = costs or TradingCosts()
    first_price = predictions[0].entry_price
    last_price = predictions[-1].exit_price

    # Gross return
    gross_return_pct = (last_price - first_price) / first_price * 100.0

    # Costs: one entry + one exit
    cost_pct = (
        costs.taker_fee_pct * 2
        + costs.spread_pct
        + costs.slippage_pct * 2
    )
    net_return_pct = gross_return_pct - cost_pct
    net_pnl = starting_capital * net_return_pct / 100.0

    # Simple drawdown from intermediate prices
    prices = [p.entry_price for p in predictions] + [predictions[-1].exit_price]
    peak = prices[0]
    max_dd = 0.0
    for p in prices:
        if p > peak:
            peak = p
        dd = (peak - p) / peak * 100.0 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    return BaselineResult(
        name="Buy & Hold",
        description="Buy BTC at start, hold until end",
        return_pct=net_return_pct,
        net_pnl=net_pnl,
        total_trades=1,
        win_rate=100.0 if net_pnl > 0 else 0.0,
        max_drawdown_pct=max_dd,
        sharpe_ratio=0.0,  # N/A for single trade
    )


# ---------------------------------------------------------------------------
# Baseline B — Random (Monte Carlo)
# ---------------------------------------------------------------------------

def random_strategy(
    predictions: list[PredictionRecord],
    costs: TradingCosts | None = None,
    risk_config: RiskConfig | None = None,
    num_simulations: int = 1000,
    seed: int | None = 42,
) -> BaselineResult:
    """Run Monte Carlo simulation of random LONG/SHORT strategy.

    For each simulation, randomly assigns LONG or SHORT to each
    prediction (50/50) and runs the full backtest with costs.

    Args:
        predictions: Sorted predictions.
        costs: Trading costs.
        risk_config: Risk configuration.
        num_simulations: Number of random simulations.
        seed: Random seed for reproducibility.

    Returns:
        BaselineResult with average metrics across all simulations.
    """
    costs = costs or TradingCosts()
    risk_config = risk_config or RiskConfig()
    rng = random.Random(seed)

    returns: list[float] = []
    win_rates: list[float] = []
    drawdowns: list[float] = []

    for _ in range(num_simulations):
        # Create modified predictions with random directions
        # We'll use a threshold of 0 (take every trade) and
        # modify predicted prices to force LONG/SHORT randomly
        random_preds = []
        for pred in predictions:
            # Randomly flip predicted direction
            if rng.random() < 0.5:
                # Force LONG: predicted = entry * 1.01
                fake_predicted = pred.entry_price * 1.01
            else:
                # Force SHORT: predicted = entry * 0.99
                fake_predicted = pred.entry_price * 0.99

            random_preds.append(PredictionRecord(
                prediction_id=pred.prediction_id,
                timestamp=pred.timestamp,
                entry_price=pred.entry_price,
                predicted_price=fake_predicted,
                predicted_min=pred.predicted_min,
                predicted_max=pred.predicted_max,
                exit_price=pred.exit_price,
                high_24h=pred.high_24h,
                low_24h=pred.low_24h,
                twap_24h=pred.twap_24h,
                prediction_method="random_baseline",
            ))

        # No threshold — take every trade
        config = StrategyConfig(
            threshold_pct=0.0,
            position_mode="independent",
            version="random_baseline",
        )

        result = run_backtest(random_preds, config, costs, risk_config)
        returns.append(result.return_pct)
        win_rates.append(result.win_rate)
        drawdowns.append(result.max_drawdown_pct)

    avg_return = sum(returns) / len(returns) if returns else 0.0
    avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 0.0
    avg_drawdown = sum(drawdowns) / len(drawdowns) if drawdowns else 0.0

    return BaselineResult(
        name="Random 50/50",
        description=f"Random LONG/SHORT ({num_simulations} simulations)",
        return_pct=avg_return,
        net_pnl=risk_config.starting_capital * avg_return / 100.0,
        total_trades=len(predictions),
        win_rate=avg_win_rate,
        max_drawdown_pct=avg_drawdown,
        sharpe_ratio=0.0,
    )


# ---------------------------------------------------------------------------
# Baseline C — Naive (price stays the same)
# ---------------------------------------------------------------------------

def naive_strategy(
    predictions: list[PredictionRecord],
    config: StrategyConfig | None = None,
    costs: TradingCosts | None = None,
    risk_config: RiskConfig | None = None,
) -> BaselineResult:
    """Naive baseline: predict price = current price.

    With any threshold > 0, this produces zero trades (NO_TRADE for all).
    With threshold = 0, direction is essentially random noise.

    Args:
        predictions: Sorted predictions.
        config: Strategy config (uses threshold).
        costs: Trading costs.
        risk_config: Risk config.

    Returns:
        BaselineResult.
    """
    config = config or StrategyConfig(threshold_pct=0.30)
    costs = costs or TradingCosts()
    risk_config = risk_config or RiskConfig()

    # Naive prediction: predicted = entry (no move expected)
    naive_preds = [
        PredictionRecord(
            prediction_id=pred.prediction_id,
            timestamp=pred.timestamp,
            entry_price=pred.entry_price,
            predicted_price=pred.entry_price,  # <-- naive
            predicted_min=pred.predicted_min,
            predicted_max=pred.predicted_max,
            exit_price=pred.exit_price,
            high_24h=pred.high_24h,
            low_24h=pred.low_24h,
            twap_24h=pred.twap_24h,
            prediction_method="naive_baseline",
        )
        for pred in predictions
    ]

    result = run_backtest(naive_preds, config, costs, risk_config)

    return BaselineResult(
        name="Naive (no change)",
        description="Predict price stays the same — 0 trades with threshold",
        return_pct=result.return_pct,
        net_pnl=result.net_pnl,
        total_trades=result.total_trades,
        win_rate=result.win_rate,
        max_drawdown_pct=result.max_drawdown_pct,
        sharpe_ratio=result.sharpe_ratio,
    )


# ---------------------------------------------------------------------------
# Baseline D — Momentum
# ---------------------------------------------------------------------------

def momentum_strategy(
    predictions: list[PredictionRecord],
    costs: TradingCosts | None = None,
    risk_config: RiskConfig | None = None,
) -> BaselineResult:
    """Momentum baseline: if price went up recently → LONG, else → SHORT.

    Uses the difference between the current entry_price and the previous
    prediction's entry_price as a momentum signal.

    Args:
        predictions: Sorted predictions (must have ≥2).
        costs: Trading costs.
        risk_config: Risk config.

    Returns:
        BaselineResult.
    """
    costs = costs or TradingCosts()
    risk_config = risk_config or RiskConfig()

    if len(predictions) < 2:
        return BaselineResult(
            name="Momentum",
            description="Follow recent trend direction",
            return_pct=0.0,
            net_pnl=0.0,
            total_trades=0,
            win_rate=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
        )

    # Build predictions where direction follows recent momentum
    momentum_preds = []
    for i in range(1, len(predictions)):
        prev_price = predictions[i - 1].entry_price
        curr_price = predictions[i].entry_price
        pred = predictions[i]

        if curr_price > prev_price:
            # Momentum up — force LONG
            fake_predicted = pred.entry_price * 1.01
        else:
            # Momentum down — force SHORT
            fake_predicted = pred.entry_price * 0.99

        momentum_preds.append(PredictionRecord(
            prediction_id=pred.prediction_id,
            timestamp=pred.timestamp,
            entry_price=pred.entry_price,
            predicted_price=fake_predicted,
            predicted_min=pred.predicted_min,
            predicted_max=pred.predicted_max,
            exit_price=pred.exit_price,
            high_24h=pred.high_24h,
            low_24h=pred.low_24h,
            twap_24h=pred.twap_24h,
            prediction_method="momentum_baseline",
        ))

    config = StrategyConfig(
        threshold_pct=0.0,  # Take every momentum signal
        position_mode="independent",
        version="momentum_baseline",
    )

    result = run_backtest(momentum_preds, config, costs, risk_config)

    return BaselineResult(
        name="Momentum",
        description="Follow recent trend direction",
        return_pct=result.return_pct,
        net_pnl=result.net_pnl,
        total_trades=result.total_trades,
        win_rate=result.win_rate,
        max_drawdown_pct=result.max_drawdown_pct,
        sharpe_ratio=result.sharpe_ratio,
    )


def run_all_baselines(
    predictions: list[PredictionRecord],
    strategy_config: StrategyConfig | None = None,
    costs: TradingCosts | None = None,
    risk_config: RiskConfig | None = None,
) -> list[BaselineResult]:
    """Run all baseline strategies and return results.

    Args:
        predictions: Sorted predictions.
        strategy_config: Used for naive baseline threshold.
        costs: Trading costs.
        risk_config: Risk config.

    Returns:
        List of BaselineResults in order [Buy&Hold, Random, Naive, Momentum].
    """
    costs = costs or TradingCosts()
    risk_config = risk_config or RiskConfig()

    results = []

    # A — Buy & Hold
    results.append(buy_and_hold_return(
        predictions, risk_config.starting_capital, costs
    ))

    # B — Random
    results.append(random_strategy(predictions, costs, risk_config))

    # C — Naive
    results.append(naive_strategy(predictions, strategy_config, costs, risk_config))

    # D — Momentum
    results.append(momentum_strategy(predictions, costs, risk_config))

    return results
