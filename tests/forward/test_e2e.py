from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from evaluation.forward.analytics import build_report, trade_report
from evaluation.forward.contracts import digest
from evaluation.forward.pipeline import generate, verify
from evaluation.forward.storage import IntegrityError, Ledger
from evaluation.trading.types import PredictionRecord, StrategyConfig

from .test_contracts import T, candles


def test_full_pipeline(tmp_path):
  legacy = Path("data/bitcoin_predictions.csv").read_bytes()
  ledger = Ledger(tmp_path / "forward.sqlite")

  def model(values):
    assert values == [100.0] * 4
    return [110.0] * 24, [90.0] * 24, [120.0] * 24

  p = generate(
    ledger,
    candles(),
    T,
    model,
    required=4,
    clock=lambda: T + timedelta(minutes=1),
    metadata={"git_sha": "fixture", "timesfm_version": "MOCK"},
  )
  original = digest(p)
  for h in range(1, 25):
    target = T + timedelta(hours=h)
    actual = "108" if h == 24 else "104"
    primary = replace(
      candles(1)[0],
      open_time=target - timedelta(hours=1),
      close_time=target,
      available_at=target,
      retrieved_at=target,
      open=actual,
      high=actual,
      low=actual,
      close=actual,
    )
    secondary = replace(primary, provider="kraken", venue="kraken")
    verify(ledger, p["prediction_id"], h, primary, secondary, target)
  assert digest(ledger.records("predictions")[0]) == original
  with pytest.raises(IntegrityError):
    verify(ledger, p["prediction_id"], 24, primary, secondary, target)
  with pytest.raises(IntegrityError):
    generate(
      ledger, candles(), T, model, required=4, clock=lambda: T + timedelta(minutes=1)
    )
  report = build_report(ledger)
  row = report["predictions"][0]
  assert row["errors"][-1]["actual"] == "108"
  assert float(row["window_statistics"]["mean_hourly_close_1_24"]) == pytest.approx(
    104 + 4 / 24
  )
  # Explicit post-generation entry quote in the deterministic market fixture.
  entry_time = T + timedelta(minutes=2)
  record = PredictionRecord(
    p["prediction_id"], entry_time.isoformat(), 100, 110, 90, 120, 108
  )
  paper = trade_report(
    [record],
    config=StrategyConfig(
      holding_period_hours=(target - entry_time).total_seconds() / 3600
    ),
  )
  trade = paper["trades"][0]
  assert trade["gross_pnl"] == 80
  assert trade["entry_fee"] == 1
  assert trade["exit_fee"] == 1.08
  assert trade["fees"] == 2.08
  assert trade["spread_cost"] == 0.52
  assert trade["slippage_cost"] == 1.04
  assert trade["net_pnl"] == pytest.approx(76.36)
  assert paper["net_pnl"] == trade["net_pnl"]
  assert paper["statistics"]["profit_factor"] == "INSUFFICIENT SAMPLE"
  assert Path("data/bitcoin_predictions.csv").read_bytes() == legacy
  assert report["legacy"]["rows"] == 12


def test_wrong_target_cannot_be_verified(tmp_path):
  ledger = Ledger(tmp_path / "forward.sqlite")
  p = generate(
    ledger,
    candles(),
    T,
    lambda values: ([110.0] * 24, [90.0] * 24, [120.0] * 24),
    required=4,
    clock=lambda: T + timedelta(minutes=1),
    metadata={"git_sha": "fixture", "timesfm_version": "MOCK"},
  )
  primary = candles(1)[0]
  secondary = replace(primary, provider="kraken", venue="kraken")
  with pytest.raises(ValueError, match="exact target"):
    verify(ledger, p["prediction_id"], 24, primary, secondary, T + timedelta(hours=24))
  assert ledger.records("verifications") == []


def test_prediction_update_and_delete_rejected(tmp_path):
  import sqlite3

  ledger = Ledger(tmp_path / "forward.sqlite")
  generate(
    ledger,
    candles(),
    T,
    lambda values: ([110.0] * 24, [90.0] * 24, [120.0] * 24),
    required=4,
    clock=lambda: T + timedelta(minutes=1),
    metadata={"git_sha": "fixture", "timesfm_version": "MOCK"},
  )
  with pytest.raises(sqlite3.IntegrityError):
    ledger.db.execute("UPDATE predictions SET payload='{}'")
  with pytest.raises(sqlite3.IntegrityError):
    ledger.db.execute("DELETE FROM predictions")
