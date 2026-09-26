"""
Trading signal generator.

Converts raw TimesFM predictions into actionable trading signals
by applying configurable threshold filters. Only generates a LONG
or SHORT signal when the predicted move exceeds the threshold,
filtering out noise, spread, and commission-level moves.

No future data is ever used — signals are based exclusively on
information available at prediction time.
"""

from __future__ import annotations

import logging

from .types import PredictionRecord, StrategyConfig, TradingSignal

logger = logging.getLogger(__name__)


def generate_signal(
    prediction: PredictionRecord,
    config: StrategyConfig,
) -> TradingSignal:
    """Generate a trading signal from a single prediction.

    The signal is LONG if the predicted return exceeds +threshold,
    SHORT if it falls below -threshold, or NO_TRADE otherwise.

    Args:
        prediction: The prediction record with entry price and predicted price.
        config: Strategy configuration with threshold.

    Returns:
        TradingSignal with direction and strength.

    Raises:
        ValueError: If entry_price is zero or negative.
    """
    if prediction.entry_price <= 0:
        raise ValueError(
            f"Invalid entry_price={prediction.entry_price} "
            f"for prediction {prediction.prediction_id}"
        )

    predicted_return_pct = (
        (prediction.predicted_price - prediction.entry_price)
        / prediction.entry_price
        * 100.0
    )
    signal_strength = abs(predicted_return_pct)
    threshold = config.threshold_pct

    if predicted_return_pct >= threshold:
        direction = "LONG"
    elif predicted_return_pct <= -threshold:
        direction = "SHORT"
    else:
        direction = "NO_TRADE"

    return TradingSignal(
        prediction_id=prediction.prediction_id,
        timestamp=prediction.timestamp,
        direction=direction,
        predicted_return_pct=predicted_return_pct,
        entry_price=prediction.entry_price,
        predicted_price=prediction.predicted_price,
        signal_strength=signal_strength,
    )


def generate_signals(
    predictions: list[PredictionRecord],
    config: StrategyConfig,
) -> list[TradingSignal]:
    """Generate trading signals for a list of predictions.

    Predictions must be sorted chronologically. This function
    applies the position_mode filter:
    - 'independent': every prediction gets its own signal.
    - 'single': skip signals while a previous trade would still be open.

    Args:
        predictions: Chronologically sorted predictions.
        config: Strategy configuration.

    Returns:
        List of TradingSignals (including NO_TRADE signals).
    """
    signals: list[TradingSignal] = []
    position_busy_until: str | None = None

    for pred in predictions:
        signal = generate_signal(pred, config)

        if config.position_mode == "single" and signal.direction != "NO_TRADE":
            # In single mode, skip if we'd still be in a position
            if position_busy_until is not None and pred.timestamp < position_busy_until:
                logger.debug(
                    "Skipping signal at %s — position busy until %s",
                    pred.timestamp,
                    position_busy_until,
                )
                # Convert to NO_TRADE
                signal = TradingSignal(
                    prediction_id=signal.prediction_id,
                    timestamp=signal.timestamp,
                    direction="NO_TRADE",
                    predicted_return_pct=signal.predicted_return_pct,
                    entry_price=signal.entry_price,
                    predicted_price=signal.predicted_price,
                    signal_strength=signal.signal_strength,
                )
            elif signal.direction != "NO_TRADE":
                # Mark position as busy for holding period
                from datetime import datetime, timedelta

                try:
                    ts = datetime.fromisoformat(
                        pred.timestamp.replace("Z", "+00:00")
                    )
                    exit_ts = ts + timedelta(hours=config.holding_period_hours)
                    position_busy_until = exit_ts.isoformat()
                except ValueError:
                    pass

        signals.append(signal)

    actionable = sum(1 for s in signals if s.direction != "NO_TRADE")
    logger.info(
        "Generated %d signals: %d actionable, %d NO_TRADE "
        "(threshold=%.2f%%)",
        len(signals),
        actionable,
        len(signals) - actionable,
        config.threshold_pct,
    )

    return signals
