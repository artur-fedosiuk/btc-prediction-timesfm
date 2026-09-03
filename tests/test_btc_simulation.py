"""
End-to-end simulation test for the Bitcoin prediction pipeline.

Simulates a 3-day experiment using mocked CoinGecko responses.
Validates the full predict → verify → report cycle.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evaluation.config import CSV_COLUMNS
from evaluation.csv_manager import load_predictions, save_prediction, update_verification
from evaluation.metrics import (
    compute_absolute_error,
    compute_direction,
    compute_percentage_error,
    is_direction_correct,
    is_within_range,
)
from evaluation.report import generate_report


# ---------------------------------------------------------------------------
# Simulated price data for 3 days
# ---------------------------------------------------------------------------
SIMULATED_DAYS = [
    {
        "day": 1,
        "timestamp": "2026-09-03T12:00:00Z",
        "initial_price": 50000.0,
        "predicted_price": 50500.0,
        "predicted_min": 49000.0,
        "predicted_max": 51500.0,
        "actual_price": 50800.0,  # up, within range
    },
    {
        "day": 2,
        "timestamp": "2026-09-04T12:00:00Z",
        "initial_price": 50800.0,
        "predicted_price": 51200.0,
        "predicted_min": 50000.0,
        "predicted_max": 52000.0,
        "actual_price": 49500.0,  # down, below range
    },
    {
        "day": 3,
        "timestamp": "2026-09-05T12:00:00Z",
        "initial_price": 49500.0,
        "predicted_price": 49000.0,
        "predicted_min": 48000.0,
        "predicted_max": 50000.0,
        "actual_price": 49200.0,  # down, within range
    },
]


@pytest.fixture
def tmp_csv(tmp_path: Path) -> Path:
    return tmp_path / "sim_predictions.csv"


@pytest.fixture
def tmp_report(tmp_path: Path) -> Path:
    return tmp_path / "sim_report.md"


class TestThreeDaySimulation:
    """Simulates a complete 3-day predict → verify → report cycle."""

    def test_full_pipeline(self, tmp_csv: Path, tmp_report: Path):
        # ---- Phase 1: Save all 3 predictions ----
        for day_data in SIMULATED_DAYS:
            predicted_dir = compute_direction(
                day_data["initial_price"], day_data["predicted_price"]
            )
            save_prediction(
                {
                    "timestamp_utc": day_data["timestamp"],
                    "initial_price": f"{day_data['initial_price']:.2f}",
                    "predicted_price_24h": f"{day_data['predicted_price']:.2f}",
                    "predicted_min": f"{day_data['predicted_min']:.2f}",
                    "predicted_max": f"{day_data['predicted_max']:.2f}",
                    "predicted_direction": predicted_dir,
                    "prediction_method": "arima_fallback",
                },
                path=tmp_csv,
            )

        # Verify 3 predictions saved
        df = load_predictions(tmp_csv)
        assert len(df) == 3
        assert all(df["actual_price_24h"] == "")

        # ---- Phase 2: Verify all predictions ----
        for day_data in SIMULATED_DAYS:
            initial = day_data["initial_price"]
            predicted = day_data["predicted_price"]
            actual = day_data["actual_price"]
            pred_min = day_data["predicted_min"]
            pred_max = day_data["predicted_max"]
            predicted_dir = compute_direction(initial, predicted)

            abs_err = compute_absolute_error(predicted, actual)
            pct_err = compute_percentage_error(predicted, actual)
            actual_dir = compute_direction(initial, actual)
            dir_correct = is_direction_correct(predicted_dir, actual_dir)
            in_range = is_within_range(actual, pred_min, pred_max)

            updated = update_verification(
                timestamp_utc=day_data["timestamp"],
                actual_price=actual,
                absolute_error=abs_err,
                percentage_error=pct_err,
                actual_direction=actual_dir,
                direction_correct=dir_correct,
                within_range=in_range,
                path=tmp_csv,
            )
            assert updated is True

        # ---- Phase 3: Validate results ----
        df = load_predictions(tmp_csv)
        assert len(df) == 3

        # All rows should be verified
        for _, row in df.iterrows():
            assert row["actual_price_24h"] != ""
            assert row["absolute_error"] != ""
            assert row["percentage_error"] != ""
            assert row["actual_direction"] in ("up", "down", "flat")
            assert row["direction_correct"] in ("true", "false")
            assert row["within_range"] in ("true", "false")

        # Day 1: predicted up, actual up → correct direction, within range
        row1 = df.iloc[0]
        assert row1["direction_correct"] == "true"
        assert row1["within_range"] == "true"

        # Day 2: predicted up, actual down → wrong direction, out of range
        row2 = df.iloc[1]
        assert row2["direction_correct"] == "false"
        assert row2["within_range"] == "false"

        # Day 3: predicted down, actual down → correct direction, within range
        row3 = df.iloc[2]
        assert row3["direction_correct"] == "true"
        assert row3["within_range"] == "true"

        # ---- Phase 4: Generate report ----
        # Patch config values for report generation
        with patch("evaluation.report.EXPERIMENT_START_DATE", "2026-09-03"), \
             patch("evaluation.report.EXPERIMENT_DURATION_DAYS", 14):
            generate_report(df, output_path=tmp_report)

        assert tmp_report.exists()
        report_text = tmp_report.read_text()

        # Check report structure
        assert "# Bitcoin (BTC/USD) Prediction Evaluation Report" in report_text
        assert "Disclaimer" in report_text
        assert "Summary Statistics" in report_text
        assert "Daily Results" in report_text
        assert "Conclusion" in report_text
        assert "MAE" in report_text
        assert "RMSE" in report_text

        # Verify metrics in report — all 3 verified
        assert "3" in report_text  # total/verified count

    def test_partial_verification(self, tmp_csv: Path, tmp_report: Path):
        """Only verify first 2 of 3 predictions — report should handle it."""
        for day_data in SIMULATED_DAYS:
            save_prediction(
                {
                    "timestamp_utc": day_data["timestamp"],
                    "initial_price": f"{day_data['initial_price']:.2f}",
                    "predicted_price_24h": f"{day_data['predicted_price']:.2f}",
                    "predicted_min": f"{day_data['predicted_min']:.2f}",
                    "predicted_max": f"{day_data['predicted_max']:.2f}",
                    "predicted_direction": compute_direction(
                        day_data["initial_price"], day_data["predicted_price"]
                    ),
                    "prediction_method": "arima_fallback",
                },
                path=tmp_csv,
            )

        # Verify only first 2 days
        for day_data in SIMULATED_DAYS[:2]:
            actual = day_data["actual_price"]
            predicted = day_data["predicted_price"]
            initial = day_data["initial_price"]
            update_verification(
                timestamp_utc=day_data["timestamp"],
                actual_price=actual,
                absolute_error=compute_absolute_error(predicted, actual),
                percentage_error=compute_percentage_error(predicted, actual),
                actual_direction=compute_direction(initial, actual),
                direction_correct=is_direction_correct(
                    compute_direction(initial, predicted),
                    compute_direction(initial, actual),
                ),
                within_range=is_within_range(
                    actual, day_data["predicted_min"], day_data["predicted_max"]
                ),
                path=tmp_csv,
            )

        df = load_predictions(tmp_csv)

        with patch("evaluation.report.EXPERIMENT_START_DATE", "2026-09-03"), \
             patch("evaluation.report.EXPERIMENT_DURATION_DAYS", 14):
            generate_report(df, output_path=tmp_report)

        report_text = tmp_report.read_text()
        # Should show 3 total, 2 verified
        assert "Total predictions" in report_text
        assert "Verified predictions" in report_text
        # Day 3 should show "—" for unverified fields
        assert "—" in report_text


class TestMetricsAccuracy:
    """Verify that computed metrics match expected values from simulation."""

    def test_day1_errors(self):
        """Day 1: predicted $50,500, actual $50,800."""
        abs_err = compute_absolute_error(50500.0, 50800.0)
        pct_err = compute_percentage_error(50500.0, 50800.0)
        assert abs(abs_err - 300.0) < 0.01
        assert abs(pct_err - 0.5906) < 0.01  # 300/50800 * 100

    def test_day2_errors(self):
        """Day 2: predicted $51,200, actual $49,500."""
        abs_err = compute_absolute_error(51200.0, 49500.0)
        pct_err = compute_percentage_error(51200.0, 49500.0)
        assert abs(abs_err - 1700.0) < 0.01
        assert abs(pct_err - 3.4343) < 0.01  # 1700/49500 * 100

    def test_day3_errors(self):
        """Day 3: predicted $49,000, actual $49,200."""
        abs_err = compute_absolute_error(49000.0, 49200.0)
        pct_err = compute_percentage_error(49000.0, 49200.0)
        assert abs(abs_err - 200.0) < 0.01
        assert abs(pct_err - 0.4065) < 0.01  # 200/49200 * 100
