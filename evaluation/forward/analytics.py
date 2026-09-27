"""Backend-only analytics. No legacy records enter forward metrics."""

import hashlib
import json
import math
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

from ..trading.backtest_engine import run_backtest
from ..trading.types import StrategyConfig
from .contracts import EXPERIMENT, digest, utc
from .storage import IntegrityError, Ledger, atomic_write

MIN_SAMPLE = 30


def trade_report(records, *, config=None, costs=None) -> dict:
  result = run_backtest(records, config or StrategyConfig(), costs)
  return {
    "accounting_mode": result.strategy_config.accounting_mode,
    "assumptions": "Paper simulation; entry and exit supplied by caller; no broker fills, borrow or funding model.",
    "trades": [asdict(t) for t in result.trades],
    "net_pnl": result.net_pnl,
    "statistics": {
      name: (
        getattr(result, name)
        if result.total_trades >= MIN_SAMPLE and math.isfinite(getattr(result, name))
        else "INSUFFICIENT SAMPLE"
      )
      for name in ("sharpe_ratio", "sortino_ratio", "profit_factor")
    },
    "statistical_status": "STATISTICALLY INCONCLUSIVE",
  }


def build_report(
  ledger: Ledger,
  *,
  legacy_path=Path("data/bitcoin_predictions.csv"),
  manifest_path=Path("data/legacy-manifest.json"),
) -> dict:
  legacy = json.loads(manifest_path.read_text())
  if hashlib.sha256(legacy_path.read_bytes()).hexdigest() != legacy["sha256"]:
    raise IntegrityError("Legacy checksum changed")
  predictions = ledger.records("predictions")
  snapshots = {digest(s): s for s in ledger.records("snapshots")}
  verifications = ledger.records("verifications")
  events = sorted(ledger.records("events"), key=lambda e: utc(e["timestamp"]))
  rows = []
  for p in predictions:
    if (
      p["input_hash"] not in snapshots
      or digest(p["forecast_points"]) != p["forecast_hash"]
    ):
      raise IntegrityError("Prediction provenance mismatch")
    own = sorted(
      [v for v in verifications if v["prediction_id"] == p["prediction_id"]],
      key=lambda v: v["horizon"],
    )
    eligible = [v for v in own if v["status"] == "VERIFIED"]
    errors = []
    for v in eligible:
      forecast = Decimal(str(p["forecast_path"][v["horizon"] - 1]))
      actual = Decimal(v["exact_price"])
      errors.append(
        {
          "horizon": v["horizon"],
          "target_timestamp": v["target_timestamp"],
          "forecast": str(forecast),
          "actual": str(actual),
          "absolute_error": str(abs(forecast - actual)),
          "absolute_percentage_error": str(abs(forecast - actual) / actual * 100),
        }
      )
    window = None
    if len(eligible) == 24:
      primary = [v["sources"][0] for v in eligible]
      window = {
        "mean_hourly_close_1_24": str(sum(Decimal(c["close"]) for c in primary) / 24),
        "twap_definition": "Equal-weight mean of 24 hourly closes; not continuous-time TWAP.",
        "high_24": str(max(Decimal(c["high"]) for c in primary)),
        "low_24": str(min(Decimal(c["low"]) for c in primary)),
      }
    rows.append(
      {
        **p,
        "verifications": own,
        "errors": errors,
        "window_statistics": window,
        "verification_status": "VERIFIED"
        if len(eligible) == 24
        else "PENDING VERIFICATION",
      }
    )
  valid = [v for v in verifications if v["status"] == "VERIFIED"]
  return {
    "schema_version": 1,
    "experiment_version": EXPERIMENT,
    "scientific_status": "INSUFFICIENT VALID DATA"
    if len(predictions) < MIN_SAMPLE
    else "STATISTICALLY INCONCLUSIVE",
    "legacy": legacy,
    "predictions": rows,
    "health": {
      "collector_status": events[-1]["status"] if events else "NOT_RUN",
      "provider_status": [e for e in events if e["operation"] == "provider_check"][-2:],
      "latest_valid_observation": max(
        (v["source_timestamp"] for v in valid), default=None
      ),
      "latest_prediction": max((p["generated_at"] for p in predictions), default=None),
      "latest_verification": max(
        (v["verified_at"] for v in verifications), default=None
      ),
      "pending_verifications": len(predictions) * 24 - len(verifications),
      "invalid_predictions": sum(
        e["operation"] == "generate" and e["status"] != "SUCCESS" for e in events
      ),
      "timesfm_status": "GENERATED" if predictions else "NOT_RUN",
      "storage_status": "INTEGRITY_CHECK_PASSED",
      "tests_build_status": "SEE_ACCEPTANCE_REPORT",
    },
    "trading": {
      "status": "NO VALID EXECUTION DATA",
      "reason": "Forecast origin precedes generation. Origin close is not an executable post-signal entry. No live portfolio result is inferred from it.",
    },
  }


def write_report(ledger: Ledger, path=Path("reports/data.json")):
  report = build_report(ledger)
  atomic_write(path, json.dumps(report, indent=2, allow_nan=False) + "\n")
  return report
