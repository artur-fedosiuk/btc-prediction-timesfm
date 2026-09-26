"""
Trading strategy evaluation package.

Provides backtesting engine, signal generation, trading metrics,
baseline strategies, and equity curve computation for evaluating
whether TimesFM predictions can produce profitable trading strategies.
"""

from .types import (
    BacktestResult,
    RiskConfig,
    StrategyConfig,
    Trade,
    TradingCosts,
    TradingSignal,
)

__all__ = [
    "BacktestResult",
    "RiskConfig",
    "StrategyConfig",
    "Trade",
    "TradingCosts",
    "TradingSignal",
]
