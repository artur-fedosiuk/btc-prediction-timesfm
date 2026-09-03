"""
Pure metric functions for the Bitcoin prediction evaluation.

All functions are stateless and deterministic — easy to unit-test.
"""

from __future__ import annotations

import math


def compute_absolute_error(predicted: float, actual: float) -> float:
    """Absolute difference between predicted and actual price."""
    return abs(predicted - actual)


def compute_percentage_error(predicted: float, actual: float) -> float:
    """Percentage error relative to the actual price.

    Returns 0.0 if actual is zero (degenerate case).
    """
    if actual == 0.0:
        return 0.0
    return abs(predicted - actual) / abs(actual) * 100.0


def compute_direction(initial: float, final: float) -> str:
    """Return 'up', 'down', or 'flat' based on price movement."""
    if final > initial:
        return "up"
    elif final < initial:
        return "down"
    return "flat"


def is_direction_correct(predicted_dir: str, actual_dir: str) -> bool:
    """Check if the predicted direction matches the actual direction."""
    return predicted_dir == actual_dir


def is_within_range(
    actual: float, min_pred: float, max_pred: float
) -> bool:
    """Check if the actual price falls within [min_pred, max_pred]."""
    return min_pred <= actual <= max_pred


def compute_mae(errors: list[float]) -> float:
    """Mean Absolute Error from a list of absolute errors.

    Returns 0.0 for an empty list.
    """
    if not errors:
        return 0.0
    return sum(errors) / len(errors)


def compute_rmse(errors: list[float]) -> float:
    """Root Mean Square Error from a list of absolute errors.

    Returns 0.0 for an empty list.
    """
    if not errors:
        return 0.0
    return math.sqrt(sum(e * e for e in errors) / len(errors))


def compute_mean_percentage_error(pct_errors: list[float]) -> float:
    """Mean percentage error from a list of percentage errors.

    Returns 0.0 for an empty list.
    """
    if not pct_errors:
        return 0.0
    return sum(pct_errors) / len(pct_errors)
