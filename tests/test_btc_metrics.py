"""
Tests for evaluation/metrics.py — pure metric functions.

Covers:
- Absolute error computation
- Percentage error computation (including zero-division edge case)
- Direction detection (up/down/flat)
- Direction correctness
- Range checking
- MAE, RMSE, and mean percentage error aggregation
"""

import math

import pytest

from evaluation.metrics import (
    compute_absolute_error,
    compute_direction,
    compute_mae,
    compute_mean_percentage_error,
    compute_percentage_error,
    compute_rmse,
    is_direction_correct,
    is_within_range,
)


# ---------------------------------------------------------------------------
# compute_absolute_error
# ---------------------------------------------------------------------------
class TestAbsoluteError:
    def test_exact_prediction(self):
        assert compute_absolute_error(100.0, 100.0) == 0.0

    def test_overestimate(self):
        assert compute_absolute_error(110.0, 100.0) == 10.0

    def test_underestimate(self):
        assert compute_absolute_error(90.0, 100.0) == 10.0

    def test_large_values(self):
        assert compute_absolute_error(50000.0, 48000.0) == 2000.0

    def test_zero_values(self):
        assert compute_absolute_error(0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# compute_percentage_error
# ---------------------------------------------------------------------------
class TestPercentageError:
    def test_exact_prediction(self):
        assert compute_percentage_error(100.0, 100.0) == 0.0

    def test_ten_percent_over(self):
        result = compute_percentage_error(110.0, 100.0)
        assert abs(result - 10.0) < 1e-9

    def test_ten_percent_under(self):
        result = compute_percentage_error(90.0, 100.0)
        assert abs(result - 10.0) < 1e-9

    def test_zero_actual(self):
        """Edge case: actual price is zero — should return 0.0."""
        assert compute_percentage_error(100.0, 0.0) == 0.0

    def test_small_error(self):
        result = compute_percentage_error(50001.0, 50000.0)
        assert abs(result - 0.002) < 0.001


# ---------------------------------------------------------------------------
# compute_direction
# ---------------------------------------------------------------------------
class TestDirection:
    def test_up(self):
        assert compute_direction(100.0, 105.0) == "up"

    def test_down(self):
        assert compute_direction(100.0, 95.0) == "down"

    def test_flat(self):
        assert compute_direction(100.0, 100.0) == "flat"

    def test_very_small_up(self):
        assert compute_direction(100.0, 100.01) == "up"


# ---------------------------------------------------------------------------
# is_direction_correct
# ---------------------------------------------------------------------------
class TestDirectionCorrect:
    def test_both_up(self):
        assert is_direction_correct("up", "up") is True

    def test_both_down(self):
        assert is_direction_correct("down", "down") is True

    def test_mismatch(self):
        assert is_direction_correct("up", "down") is False

    def test_flat_vs_up(self):
        assert is_direction_correct("flat", "up") is False


# ---------------------------------------------------------------------------
# is_within_range
# ---------------------------------------------------------------------------
class TestWithinRange:
    def test_inside(self):
        assert is_within_range(105.0, 100.0, 110.0) is True

    def test_at_lower_bound(self):
        assert is_within_range(100.0, 100.0, 110.0) is True

    def test_at_upper_bound(self):
        assert is_within_range(110.0, 100.0, 110.0) is True

    def test_below_range(self):
        assert is_within_range(99.0, 100.0, 110.0) is False

    def test_above_range(self):
        assert is_within_range(111.0, 100.0, 110.0) is False


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------
class TestAggregateMetrics:
    def test_mae_basic(self):
        errors = [10.0, 20.0, 30.0]
        assert abs(compute_mae(errors) - 20.0) < 1e-9

    def test_mae_empty(self):
        assert compute_mae([]) == 0.0

    def test_mae_single(self):
        assert compute_mae([5.0]) == 5.0

    def test_rmse_basic(self):
        errors = [3.0, 4.0]
        # RMSE = sqrt((9 + 16) / 2) = sqrt(12.5) ≈ 3.5355
        expected = math.sqrt(12.5)
        assert abs(compute_rmse(errors) - expected) < 1e-4

    def test_rmse_empty(self):
        assert compute_rmse([]) == 0.0

    def test_rmse_single(self):
        assert compute_rmse([7.0]) == 7.0

    def test_rmse_all_equal(self):
        errors = [5.0, 5.0, 5.0]
        assert abs(compute_rmse(errors) - 5.0) < 1e-9

    def test_mean_pct_error(self):
        pct = [2.0, 4.0, 6.0]
        assert abs(compute_mean_percentage_error(pct) - 4.0) < 1e-9

    def test_mean_pct_error_empty(self):
        assert compute_mean_percentage_error([]) == 0.0
