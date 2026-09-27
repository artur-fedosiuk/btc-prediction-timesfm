"""
Tests for evaluation/csv_manager.py — CSV load/save/update.

Uses temporary files to verify:
- CSV creation with correct headers
- Prediction row appending
- Verification update without losing existing data
- Multiple predictions in sequence
"""

import csv
from pathlib import Path

import pandas as pd
import pytest

from evaluation.config import CSV_COLUMNS
from evaluation.csv_manager import (
    load_predictions,
    save_prediction,
    update_verification,
)


@pytest.fixture
def tmp_csv(tmp_path: Path) -> Path:
    """Return a path to a temporary CSV file."""
    return tmp_path / "test_predictions.csv"


class TestLoadPredictions:
    def test_creates_file_if_missing(self, tmp_csv: Path):
        df = load_predictions(tmp_csv)
        assert tmp_csv.exists()
        assert list(df.columns) == CSV_COLUMNS
        assert len(df) == 0

    def test_loads_existing_data(self, tmp_csv: Path):
        # Create a CSV with one row
        with open(tmp_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp_utc",
                    "initial_price",
                    "predicted_price_24h",
                    "predicted_min",
                    "predicted_max",
                    "actual_price_24h",
                    "absolute_error",
                    "percentage_error",
                    "predicted_direction",
                    "actual_direction",
                    "direction_correct",
                    "within_range",
                    "prediction_method",
                ]
            )
            writer.writerow(
                [
                    "2026-09-03T12:00:00Z",
                    "50000.00",
                    "51000.00",
                    "49000.00",
                    "52000.00",
                    "",
                    "",
                    "",
                    "up",
                    "",
                    "",
                    "",
                    "arima_fallback",
                ]
            )
        df = load_predictions(tmp_csv)
        assert len(df) == 1
        assert df.iloc[0]["initial_price"] == "50000.00"


class TestSavePrediction:
    def test_append_single_row(self, tmp_csv: Path):
        save_prediction(
            {
                "timestamp_utc": "2026-09-03T12:00:00Z",
                "initial_price": "50000.00",
                "predicted_price_24h": "51000.00",
                "predicted_min": "49000.00",
                "predicted_max": "52000.00",
                "predicted_direction": "up",
                "prediction_method": "arima_fallback",
            },
            path=tmp_csv,
        )
        df = load_predictions(tmp_csv)
        assert len(df) == 1
        assert df.iloc[0]["timestamp_utc"] == "2026-09-03T12:00:00Z"

    def test_append_multiple_rows(self, tmp_csv: Path):
        for day in range(3):
            save_prediction(
                {
                    "timestamp_utc": f"2026-09-0{day + 3}T12:00:00Z",
                    "initial_price": f"{50000 + day * 100:.2f}",
                    "predicted_price_24h": f"{50100 + day * 100:.2f}",
                    "predicted_min": f"{49900 + day * 100:.2f}",
                    "predicted_max": f"{50200 + day * 100:.2f}",
                    "predicted_direction": "up",
                    "prediction_method": "arima_fallback",
                },
                path=tmp_csv,
            )
        df = load_predictions(tmp_csv)
        assert len(df) == 3

    def test_never_overwrites_existing(self, tmp_csv: Path):
        save_prediction(
            {
                "timestamp_utc": "2026-09-03T12:00:00Z",
                "initial_price": "50000.00",
                "predicted_price_24h": "51000.00",
                "prediction_method": "arima_fallback",
            },
            path=tmp_csv,
        )
        save_prediction(
            {
                "timestamp_utc": "2026-09-04T12:00:00Z",
                "initial_price": "51000.00",
                "predicted_price_24h": "52000.00",
                "prediction_method": "arima_fallback",
            },
            path=tmp_csv,
        )
        df = load_predictions(tmp_csv)
        assert len(df) == 2
        assert df.iloc[0]["initial_price"] == "50000.00"
        assert df.iloc[1]["initial_price"] == "51000.00"


class TestUpdateVerification:
    def test_update_existing_row(self, tmp_csv: Path):
        save_prediction(
            {
                "timestamp_utc": "2026-09-03T12:00:00Z",
                "initial_price": "50000.00",
                "predicted_price_24h": "51000.00",
                "predicted_min": "49000.00",
                "predicted_max": "52000.00",
                "predicted_direction": "up",
                "prediction_method": "arima_fallback",
            },
            path=tmp_csv,
        )
        result = update_verification(
            timestamp_utc="2026-09-03T12:00:00Z",
            actual_price=51500.0,
            absolute_error=500.0,
            percentage_error=0.98,
            actual_direction="up",
            direction_correct=True,
            within_range=True,
            path=tmp_csv,
        )
        assert result is True
        df = load_predictions(tmp_csv)
        assert df.iloc[0]["actual_price_24h"] == "51500.0"
        assert df.iloc[0]["direction_correct"] == "true"
        assert df.iloc[0]["within_range"] == "true"

    def test_update_nonexistent_returns_false(self, tmp_csv: Path):
        # Ensure file exists with headers
        load_predictions(tmp_csv)
        result = update_verification(
            timestamp_utc="2099-01-01T00:00:00Z",
            actual_price=100.0,
            absolute_error=0.0,
            percentage_error=0.0,
            actual_direction="flat",
            direction_correct=False,
            within_range=False,
            path=tmp_csv,
        )
        assert result is False

    def test_preserves_other_rows(self, tmp_csv: Path):
        for day in [3, 4, 5]:
            save_prediction(
                {
                    "timestamp_utc": f"2026-09-0{day}T12:00:00Z",
                    "initial_price": f"{50000 + day * 100:.2f}",
                    "predicted_price_24h": f"{50100 + day * 100:.2f}",
                    "predicted_min": f"{49900 + day * 100:.2f}",
                    "predicted_max": f"{50300 + day * 100:.2f}",
                    "predicted_direction": "up",
                    "prediction_method": "arima_fallback",
                },
                path=tmp_csv,
            )
        # Update only day 4
        update_verification(
            timestamp_utc="2026-09-04T12:00:00Z",
            actual_price=50600.0,
            absolute_error=100.0,
            percentage_error=0.2,
            actual_direction="up",
            direction_correct=True,
            within_range=True,
            path=tmp_csv,
        )
        df = load_predictions(tmp_csv)
        assert len(df) == 3
        # Day 3 and 5 should remain unverified
        assert df.iloc[0]["actual_price_24h"] == ""
        assert df.iloc[2]["actual_price_24h"] == ""
        # Day 4 should be verified
        assert df.iloc[1]["actual_price_24h"] == "50600.0"


def test_refuse_legacy_append(tmp_path):
    source = Path("data/bitcoin_predictions.csv").read_bytes()
    path = tmp_path / "legacy.csv"
    path.write_bytes(source)
    with pytest.raises(ValueError, match="schema mismatch"):
        save_prediction({"timestamp_utc": "2026-09-27T12:00:00Z"}, path)
    assert path.read_bytes() == source


def test_archive_read_only():
    with pytest.raises(ValueError, match="read-only"):
        save_prediction({"timestamp_utc": "2026-09-27T12:00:00Z"})
