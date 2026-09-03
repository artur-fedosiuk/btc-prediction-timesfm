"""
Predictor module: runs TimesFM 3.0 or ARIMA fallback.

- `predict_with_timesfm()` loads the real model (requires torch + timesfm3).
- `predict_with_fallback()` uses statsmodels ARIMA — lightweight for CI.

Both return the same schema so the rest of the pipeline is method-agnostic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Standardised prediction output."""

    forecast: float  # Predicted price after 24h
    min_price: float  # Lower bound (q10 or confidence interval)
    max_price: float  # Upper bound (q90 or confidence interval)
    method: str  # "timesfm" or "arima_fallback"


def predict_with_timesfm(history: np.ndarray) -> PredictionResult:
    """Run TimesFM 3.0 for a 1-step-ahead forecast with quantiles.

    Requires torch and timesfm3 to be installed. Reuses the same pattern
    as the chat_app/app.py Flask backend.

    Args:
        history: 1-D array of recent BTC/USD prices (hourly, ≥30 points).

    Returns:
        PredictionResult with the median forecast and q10/q90 bounds.
    """
    try:
        import torch
        from timesfm3 import ModelConfig, TimesFM3Evaluator
    except ImportError as exc:
        raise RuntimeError(
            "TimesFM 3.0 dependencies not available. "
            "Install with: pip install timesfm[torch]\n"
            f"Original error: {exc}"
        ) from exc

    # Pick best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info("Loading TimesFM 3.0 on device=%s ...", device)
    config = ModelConfig(
        checkpoint_path="google/timesfm-3.0-pytorch",
        per_core_batch_size=4,
        device=device,
    )
    forecaster = TimesFM3Evaluator(config)

    # Use last 512 points max (same as chat_app)
    context = history[-min(512, len(history)) :].astype(np.float32)

    outputs = list(
        forecaster.predict_batch(
            [context],
            horizon=1,  # 1 step = 24h if hourly data aggregated to daily
            return_quantiles=True,
            use_symmetric_averaging=True,
        )
    )
    result = outputs[0]

    # result.forecast shape: (1,)  — median
    # result.quantiles shape: (1, 9)  — q10 to q90
    forecast_val = float(result.forecast[0])
    q10_val = float(result.quantiles[0, 0])  # 0.1 quantile
    q90_val = float(result.quantiles[0, 8])  # 0.9 quantile

    logger.info(
        "TimesFM prediction: $%.2f [min=$%.2f, max=$%.2f]",
        forecast_val,
        q10_val,
        q90_val,
    )
    return PredictionResult(
        forecast=forecast_val,
        min_price=q10_val,
        max_price=q90_val,
        method="timesfm",
    )


def predict_with_fallback(history: np.ndarray) -> PredictionResult:
    """Lightweight ARIMA fallback for GitHub Actions CI.

    Uses statsmodels ARIMA(5,1,0) on recent price data to produce
    a 1-step-ahead forecast with 80% confidence interval.

    Args:
        history: 1-D array of recent BTC/USD prices (≥30 points).

    Returns:
        PredictionResult with ARIMA forecast and confidence bounds.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:
        raise RuntimeError(
            "statsmodels not available. "
            "Install with: pip install statsmodels\n"
            f"Original error: {exc}"
        ) from exc

    # Use last 168 hourly points (7 days) for fitting
    data = history[-min(168, len(history)) :].astype(np.float64)

    logger.info("Running ARIMA(5,1,0) fallback on %d data points...", len(data))

    model = ARIMA(data, order=(5, 1, 0))
    fitted = model.fit()

    # Forecast 1 step ahead with 80% confidence interval (q10–q90 equivalent)
    forecast_result = fitted.get_forecast(steps=1)
    forecast_val = float(forecast_result.predicted_mean[0])
    conf_int = forecast_result.conf_int(alpha=0.20)  # 80% CI
    min_val = float(conf_int[0, 0])
    max_val = float(conf_int[0, 1])

    logger.info(
        "ARIMA fallback prediction: $%.2f [min=$%.2f, max=$%.2f]",
        forecast_val,
        min_val,
        max_val,
    )
    return PredictionResult(
        forecast=forecast_val,
        min_price=min_val,
        max_price=max_val,
        method="arima_fallback",
    )
