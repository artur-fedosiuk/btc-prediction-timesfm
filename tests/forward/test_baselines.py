"""Unit tests for evaluation/forward/baselines.py.

Tests are deliberately lightweight: no external dependencies (statsmodels
optional), no SQLite, no network.  They verify:

1.  Path lengths and types for all three baselines.
2.  Persistence baseline is constant.
3.  Drift baseline trend direction.
4.  ARIMA baseline falls back gracefully when statsmodels is missing.
5.  Metrics are computed on log-returns (scale-free).
6.  Direction accuracy is 1.0 for a perfect forecast.
7.  report_baselines minimum-sample gate.
"""

from __future__ import annotations

import importlib
import math
import sys
import unittest
from unittest.mock import patch


class TestPaths(unittest.TestCase):
    """Baseline path shapes and types."""

    def setUp(self):
        from evaluation.forward.baselines import (
            _arima_path,
            _drift_path,
            _persistence_path,
        )
        self._pers  = _persistence_path
        self._drift = _drift_path
        self._arima = _arima_path

    def _flat(self, n=200):
        return [84_000.0] * n

    def _rising(self, n=200):
        return [84_000.0 + i * 10 for i in range(n)]

    # ---- persistence -------------------------------------------------------

    def test_persistence_length(self):
        path = self._pers(self._flat(), horizon=24)
        self.assertEqual(len(path), 24)

    def test_persistence_constant(self):
        values = self._rising()
        path = self._pers(values, horizon=24)
        last = values[-1]
        self.assertTrue(all(p == last for p in path))

    def test_persistence_single_point(self):
        path = self._pers([42.0], horizon=3)
        self.assertEqual(path, [42.0, 42.0, 42.0])

    # ---- drift -------------------------------------------------------------

    def test_drift_length(self):
        path = self._drift(self._rising(), horizon=24)
        self.assertEqual(len(path), 24)

    def test_drift_exact_sequence(self):
        """Series 0..99 must give [100, 101, 102] — anchored to last value."""
        values = list(range(100))
        path = self._drift([float(v) for v in values], horizon=3)
        self.assertEqual(path, [100.0, 101.0, 102.0])

    def test_drift_constant_flat(self):
        """Constant series → mean_diff=0 → path repeats the last value."""
        values = [42.0] * 50
        path = self._drift(values, horizon=5)
        self.assertEqual(path, [42.0] * 5)

    def test_drift_single_point(self):
        """Fewer than 2 points → falls back to persistence."""
        path = self._drift([77.0], horizon=3)
        self.assertEqual(path, [77.0, 77.0, 77.0])

    def test_drift_direction_up(self):
        path = self._drift(self._rising(), horizon=24)
        # Each step should be greater than the previous (positive slope).
        self.assertTrue(all(path[i] < path[i + 1] for i in range(len(path) - 1)))

    def test_drift_direction_down(self):
        values = [84_000.0 - i * 10 for i in range(200)]
        path = self._drift(values, horizon=24)
        self.assertTrue(all(path[i] > path[i + 1] for i in range(len(path) - 1)))

    def test_drift_flat(self):
        """Flat series → slope ≈ 0 → forecast ≈ last value."""
        path = self._drift(self._flat(), horizon=24)
        last = self._flat()[-1]
        for p in path:
            self.assertAlmostEqual(p, last, delta=1.0)

    def test_drift_min_clamp(self):
        """Drift never returns zero or negative price."""
        values = [100.0 - i * 10 for i in range(20)]  # hits zero
        path = self._drift(values, horizon=24)
        self.assertTrue(all(p > 0 for p in path))

    # ---- ARIMA -------------------------------------------------------------

    def test_arima_length(self):
        path, _fb = self._arima(self._rising(), horizon=24)
        self.assertEqual(len(path), 24)

    def test_arima_fallback_flag(self):
        """_arima_path returns (path, fallback=True) when statsmodels absent."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "statsmodels.tsa.arima.model":
                raise ImportError("no statsmodels")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            from evaluation.forward import baselines as bl
            path, fallback = bl._arima_path(self._rising(), horizon=24)
        self.assertEqual(len(path), 24)
        self.assertTrue(fallback)
        self.assertTrue(all(p > 0 for p in path))

    def test_arima_fallback_without_statsmodels(self):
        """When statsmodels is unavailable, ARIMA silently falls back to drift."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "statsmodels.tsa.arima.model":
                raise ImportError("no statsmodels")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            # Re-execute arima path to exercise the except branch.
            from evaluation.forward import baselines as bl
            path, _fb = bl._arima_path(self._rising(), horizon=24)
        self.assertEqual(len(path), 24)
        self.assertTrue(all(p > 0 for p in path))

    def test_arima_min_clamp(self):
        path, _fb = self._arima(self._flat(10), horizon=24)  # short series → drift fallback
        self.assertTrue(all(p > 0 for p in path))


class TestMetrics(unittest.TestCase):
    """Log-return metrics and direction accuracy."""

    def _run(self, origin, path, actuals_dict):
        from evaluation.forward.baselines import _compute_metrics
        return _compute_metrics(origin, path, actuals_dict)

    def test_perfect_forecast_zero_mae(self):
        origin = 84_000.0
        path = [84_100.0] * 24
        actuals = {h: 84_100.0 for h in range(1, 25)}
        m = self._run(origin, path, actuals)
        self.assertAlmostEqual(m["mae_log_return"], 0.0, places=8)

    def test_perfect_forecast_direction_accuracy_one(self):
        origin = 84_000.0
        path = [84_100.0] * 24
        actuals = {h: 84_200.0 for h in range(1, 25)}  # both up
        m = self._run(origin, path, actuals)
        self.assertAlmostEqual(m["direction_accuracy"], 1.0, places=4)

    def test_opposite_direction_zero_accuracy(self):
        origin = 84_000.0
        # forecast goes up, actual goes down
        path = [85_000.0] * 24
        actuals = {h: 83_000.0 for h in range(1, 25)}
        m = self._run(origin, path, actuals)
        self.assertAlmostEqual(m["direction_accuracy"], 0.0, places=4)

    def test_persistence_actual_up_direction_none(self):
        """Persistence (lr_forecast=0) has no directional opinion → excluded."""
        origin = 84_000.0
        path = [origin] * 24       # zero log-return → no opinion
        actuals = {h: 85_000.0 for h in range(1, 25)}  # actual goes up
        m = self._run(origin, path, actuals)
        self.assertIsNone(m["direction_accuracy"])

    def test_persistence_actual_down_direction_none(self):
        """Persistence (lr_forecast=0) vs actual down → still None."""
        origin = 84_000.0
        path = [origin] * 24
        actuals = {h: 83_000.0 for h in range(1, 25)}  # actual goes down
        m = self._run(origin, path, actuals)
        self.assertIsNone(m["direction_accuracy"])

    def test_both_zero_log_return_direction_none(self):
        """Both forecast and actual equal origin → no directional opinion → None."""
        origin = 84_000.0
        path = [origin] * 24
        actuals = {h: origin for h in range(1, 25)}
        m = self._run(origin, path, actuals)
        self.assertIsNone(m["direction_accuracy"])

    def test_mixed_direction_denominator(self):
        """dir_acc denominator = horizons with lr_forecast != 0, not all verified."""
        origin = 84_000.0
        # First 12 horizons: forecast == origin (no opinion → excluded).
        # Last 12 horizons: forecast up, actual up → 12 correct out of 12.
        path = [origin] * 12 + [85_000.0] * 12
        actuals = {h: 85_000.0 for h in range(1, 25)}
        m = self._run(origin, path, actuals)
        # n_verified = 24 (all horizons have MAE), but direction uses 12.
        self.assertEqual(m["n_verified"], 24)
        self.assertAlmostEqual(m["direction_accuracy"], 1.0, places=4)

    def test_mae_is_scale_free(self):
        """MAE on log returns must be the same for doubled prices."""
        origin1, path1 = 100.0, [110.0] * 24
        actuals1 = {h: 105.0 for h in range(1, 25)}
        origin2, path2 = 200.0, [220.0] * 24
        actuals2 = {h: 210.0 for h in range(1, 25)}
        from evaluation.forward.baselines import _compute_metrics
        m1 = _compute_metrics(origin1, path1, actuals1)
        m2 = _compute_metrics(origin2, path2, actuals2)
        self.assertAlmostEqual(m1["mae_log_return"], m2["mae_log_return"], places=6)

    def test_no_actuals_returns_note(self):
        from evaluation.forward.baselines import _compute_metrics
        m = _compute_metrics(84_000.0, [84_100.0] * 24, {})
        self.assertIn("note", m)

    def test_n_verified_count(self):
        origin = 84_000.0
        path = [84_100.0] * 24
        actuals = {h: 84_050.0 for h in range(1, 13)}  # only first 12
        m = self._run(origin, path, actuals)
        self.assertEqual(m["n_verified"], 12)

    def test_per_horizon_fields(self):
        origin = 84_000.0
        path = [84_100.0] * 24
        actuals = {1: 84_200.0}
        m = self._run(origin, path, actuals)
        ph = [p for p in m["per_horizon"] if p["horizon"] == 1][0]
        self.assertIn("lr_forecast", ph)
        self.assertIn("lr_actual", ph)
        self.assertIn("log_ae", ph)


class TestBaselinesForPrediction(unittest.TestCase):
    """Integration: baselines_for_prediction returns all four models."""

    def _sample_values(self, n=512):
        return [84_000.0 + math.sin(i / 10) * 100 for i in range(n)]

    def test_keys_present(self):
        from evaluation.forward.baselines import baselines_for_prediction
        vals = self._sample_values()
        fp = [84_100.0 + i for i in range(24)]
        result = baselines_for_prediction(vals, fp, [])
        self.assertIn("persistence", result)
        self.assertIn("drift", result)
        self.assertIn("arima", result)
        self.assertIn("timesfm", result)

    def test_path_lengths(self):
        from evaluation.forward.baselines import baselines_for_prediction
        vals = self._sample_values()
        fp = [84_100.0] * 24
        result = baselines_for_prediction(vals, fp, [])
        for name, data in result.items():
            self.assertEqual(len(data["path"]), 24, msg=f"{name} path length")

    def test_origin_price(self):
        from evaluation.forward.baselines import baselines_for_prediction
        vals = self._sample_values()
        fp = [84_100.0] * 24
        result = baselines_for_prediction(vals, fp, [])
        for name, data in result.items():
            self.assertAlmostEqual(data["origin_price"], vals[-1], places=4)

    def test_with_verifications(self):
        from evaluation.forward.baselines import baselines_for_prediction
        vals = self._sample_values()
        fp = [84_100.0 + i for i in range(24)]
        vfs = [
            {"status": "VERIFIED", "horizon": h, "exact_price": str(84_050.0 + h * 2)}
            for h in range(1, 13)
        ]
        result = baselines_for_prediction(vals, fp, vfs)
        for name, data in result.items():
            m = data["metrics"]
            if "n_verified" in m:
                self.assertEqual(m["n_verified"], 12)

    def test_empty_input(self):
        from evaluation.forward.baselines import baselines_for_prediction
        result = baselines_for_prediction([], [], [])
        self.assertEqual(result, {})


class TestReportBaselines(unittest.TestCase):
    """report_baselines gate and structure."""

    def _make_prediction(self, pid, input_values, forecast_path, forecast_origin=""):
        d = {
            "prediction_id": pid,
            "_input_values": input_values,
            "forecast_path": forecast_path,
        }
        if forecast_origin:
            d["forecast_origin"] = forecast_origin
        return d

    def test_insufficient_sample_status(self):
        from evaluation.forward.baselines import report_baselines
        # 10 predictions on the same day → only 1 distinct UTC date.
        preds = [
            self._make_prediction(
                f"p{i}",
                [84_000.0 + j for j in range(512)],
                [84_100.0] * 24,
                forecast_origin=f"2026-01-01T{i:02d}:00:00+00:00",
            )
            for i in range(10)
        ]
        result = report_baselines(preds, [])
        self.assertEqual(result["status"], "STATISTICALLY_INCONCLUSIVE")
        self.assertEqual(result["n_daily_origins"], 1)  # 1 distinct date

    def test_n_daily_origins_counts_distinct_dates(self):
        """Multiple predictions on distinct dates → each date counts once."""
        from evaluation.forward.baselines import report_baselines
        preds = [
            self._make_prediction(
                f"p{i}",
                [84_000.0 + j for j in range(512)],
                [84_100.0] * 24,
                forecast_origin=f"2026-01-{i+1:02d}T12:00:00+00:00",
            )
            for i in range(5)
        ]
        result = report_baselines(preds, [])
        self.assertEqual(result["n_daily_origins"], 5)

    def test_sufficient_sample_status(self):
        from evaluation.forward.baselines import report_baselines, MIN_DAILY_ORIGINS
        # 60 predictions, each on a distinct date → 60 daily origins.
        preds = [
            self._make_prediction(
                f"p{i}",
                [84_000.0 + j for j in range(512)],
                [84_100.0] * 24,
                forecast_origin=f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}T12:00:00+00:00",
            )
            for i in range(MIN_DAILY_ORIGINS)
        ]
        result = report_baselines(preds, [])
        self.assertEqual(result["status"], "EVALUABLE")

    def test_summary_keys(self):
        from evaluation.forward.baselines import report_baselines
        preds = [
            self._make_prediction(
                "p0",
                [84_000.0 + j for j in range(512)],
                [84_100.0] * 24,
            )
        ]
        result = report_baselines(preds, [])
        for model in ("persistence", "drift", "arima", "timesfm"):
            self.assertIn(model, result["summary"])

    def test_arima_fallback_count_in_report(self):
        from evaluation.forward.baselines import report_baselines
        preds = [
            self._make_prediction(
                "p0",
                [84_000.0 + j for j in range(512)],
                [84_100.0] * 24,
            )
        ]
        result = report_baselines(preds, [])
        self.assertIn("arima_fallback_count", result)
        self.assertIsInstance(result["arima_fallback_count"], int)

    def test_no_input_values_flagged(self):
        from evaluation.forward.baselines import report_baselines
        preds = [{"prediction_id": "p0", "_input_values": [], "forecast_path": []}]
        result = report_baselines(preds, [])
        self.assertIn("note", result["per_prediction"][0])

    def test_schema_version(self):
        from evaluation.forward.baselines import report_baselines
        result = report_baselines([], [])
        self.assertEqual(result["schema_version"], 1)
        self.assertIn("metric_definition", result)

    def test_n_direction_in_metrics(self):
        """Per-prediction metrics must include n_direction."""
        from evaluation.forward.baselines import baselines_for_prediction
        origin = 84_000.0
        vals = [origin - 511 + j for j in range(512)]
        fp = [origin + 100.0] * 24  # all up
        vfs = [
            {"status": "VERIFIED", "horizon": h, "exact_price": str(origin + 200.0)}
            for h in range(1, 25)
        ]
        result = baselines_for_prediction(vals, fp, vfs)
        m = result["timesfm"]["metrics"]
        self.assertIn("n_direction", m)
        self.assertEqual(m["n_direction"], 24)

    def test_aggregate_direction_uses_n_direction(self):
        """Aggregated direction must weight by n_direction, not n_verified.

        Prediction A: forecast all UP, actual all DOWN → 0/24 correct.
        Prediction B: 12 flat (excluded) + 12 UP, actual UP → 12/12 correct.
        Correct aggregate for timesfm: 12 correct / 36 directional = 0.3333.
        With the old bug (n_verified): (0*24 + 1*24)/(24+24) = 0.5.
        """
        from evaluation.forward.baselines import report_baselines
        origin = 84_000.0
        vals = [origin - 511 + j for j in range(512)]

        # A: all up, actual all down.
        fp_a = [origin + 500.0] * 24
        vfs_a = [
            {"status": "VERIFIED", "horizon": h, "exact_price": str(origin - 500.0),
             "prediction_id": "A"}
            for h in range(1, 25)
        ]
        # B: first 12 = origin (flat, excluded from direction), last 12 = up, actual up.
        fp_b = [origin] * 12 + [origin + 500.0] * 12
        vfs_b = [
            {"status": "VERIFIED", "horizon": h, "exact_price": str(origin + 500.0),
             "prediction_id": "B"}
            for h in range(1, 25)
        ]

        preds = [
            {"prediction_id": "A", "_input_values": vals, "forecast_path": fp_a},
            {"prediction_id": "B", "_input_values": vals, "forecast_path": fp_b},
        ]
        result = report_baselines(preds, vfs_a + vfs_b)
        tfm = result["summary"]["timesfm"]
        # 12 correct out of 36 directional horizons = 0.3333.
        self.assertAlmostEqual(tfm["direction_accuracy"], 12 / 36, places=4)

    def test_lr_actual_zero_excluded_from_direction(self):
        """When lr_actual == 0, that horizon has no directional move → excluded."""
        from evaluation.forward.baselines import _compute_metrics
        origin = 84_000.0
        path = [origin + 100.0] * 24  # forecast up
        # All actuals equal origin → lr_actual == 0 → excluded.
        actuals = {h: origin for h in range(1, 25)}
        m = _compute_metrics(origin, path, actuals)
        self.assertIsNone(m["direction_accuracy"])
        self.assertEqual(m["n_verified"], 24)
        self.assertEqual(m["n_direction"], 0)


if __name__ == "__main__":
    unittest.main()
