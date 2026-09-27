"""Pure orchestration boundaries; network and model providers are injected."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import struct
import subprocess
import uuid
from datetime import datetime, timezone
from decimal import Decimal

from .contracts import (
  CHECKPOINT,
  EXPERIMENT,
  HOUR,
  REVISION,
  Candle,
  DataError,
  ForecastValidator,
  MarketDataValidator,
  digest,
  utc,
)
from .storage import IntegrityError, Ledger


def now():
  return datetime.now(timezone.utc)


def environment(device: str) -> dict:
  versions = {}
  for name in ("timesfm", "torch", "numpy", "pandas", "statsmodels"):
    try:
      versions[name + "_version"] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
      versions[name + "_version"] = "NOT_INSTALLED"
  return {
    "python_version": platform.python_version(),
    "os": platform.platform(),
    "architecture": platform.machine(),
    "github": {
      key: os.getenv(key)
      for key in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_SHA", "GITHUB_REF")
    },
    **versions,
    "device": device,
    "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "working_tree_dirty": bool(
      subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    ),
  }


def record_event(ledger: Ledger, status: str, message: str, *, operation: str) -> dict:
  event = {
    "id": str(uuid.uuid4()),
    "timestamp": now().isoformat(),
    "status": status,
    "message": message,
    "operation": operation,
  }
  with ledger.db:
    ledger.add("events", event["id"], event)
  return event


def generate(
  ledger: Ledger,
  candles: list[Candle],
  cutoff: datetime,
  model,
  *,
  clock=now,
  required=512,
  metadata=None,
) -> dict:
  origin = MarketDataValidator.validate(candles, cutoff, required)
  identity = digest(
    {"experiment": EXPERIMENT, "origin": origin.isoformat(), "revision": REVISION}
  )
  if any(p["prediction_id"] == identity for p in ledger.records("predictions")):
    raise IntegrityError("Duplicate prediction ID/origin")
  # Capture exactly the float32 values submitted to the model, with the time axis.
  values = [struct.unpack("<f", struct.pack("<f", float(c.close)))[0] for c in candles]
  snapshot = {
    "schema_version": 1,
    "candles": [c.to_dict() for c in candles],
    "input_dtype": "float32",
    "input_series": "hourly BTC/USD close",
    "input_values": values,
    "timestamps": [utc(c.close_time).isoformat() for c in candles],
  }
  input_hash = digest(snapshot)
  path, lower, upper = model(values)
  points = ForecastValidator.validate(path, lower, upper, origin)
  generated = utc(clock())
  if not utc(cutoff) <= generated < origin + HOUR:
    raise DataError(
      "INVALID_FORECAST", "Generation must finish after cutoff and before first target"
    )
  manifest = metadata or environment("cpu")
  prediction = {
    "schema_version": 1,
    "prediction_id": identity,
    "experiment_id": EXPERIMENT,
    "experiment_version": EXPERIMENT,
    "generated_at": generated.isoformat(),
    "data_cutoff": utc(cutoff).isoformat(),
    "forecast_origin": origin.isoformat(),
    "provider": candles[0].provider,
    "venue": candles[0].venue,
    "symbol": "BTC/USD",
    "quote_currency": "USD",
    "requested_model": "TimesFM",
    "actual_model_used": "TimesFM",
    "fallback_used": False,
    "fallback_reason": None,
    "model_version": manifest.get("timesfm_version", "UNKNOWN"),
    "checkpoint": CHECKPOINT,
    "checkpoint_revision": REVISION,
    "input_snapshot_ref": input_hash,
    "input_hash": input_hash,
    "forecast_hash": digest(points),
    "forecast_horizon": 24,
    "forecast_frequency": "1h",
    "forecast_points": points,
    "forecast_path": list(map(float, path)),
    "lower_path": list(map(float, lower)),
    "upper_path": list(map(float, upper)),
    "status": "VALID",
    "manifest": {
      **manifest,
      **getattr(model, "metadata", {}),
      "forecast_config": {
        "horizon": 24,
        "context_length": required,
        "symmetric_averaging": True,
        "make_positive": True,
        "sort_quantiles": True,
        "use_znorm": False,
        "padding_mode": "none",
        "univariate": False,
        "batch_size": 1,
        "seed": 0,
        "threads": 2,
        "input_dtype": "float32",
        "quantiles": "nominal 80% pointwise interval; uncalibrated on BTC",
      },
      "provider_config": {
        "primary": candles[0].provider,
        "symbol": "BTC/USD",
        "frequency": "1h",
      },
    },
  }
  experiment = {
    "experiment_id": EXPERIMENT,
    "experiment_version": EXPERIMENT,
    "schema_version": 1,
    "created_at": generated.isoformat(),
    "forecast_frequency": "1h",
    "forecast_horizon": 24,
    "model_name": "TimesFM",
    "model_version": prediction["model_version"],
    "strategy_version": "diagnostic_v2",
    "git_commit": manifest["git_sha"],
  }
  with ledger.db:
    if not ledger.records("experiments"):
      ledger.add("experiments", EXPERIMENT, experiment)
    # Distinct origins cannot share a complete timestamped snapshot.
    ledger.add("snapshots", input_hash, snapshot)
    ledger.add("predictions", identity, prediction)
  return prediction


def verify(
  ledger: Ledger,
  prediction_id: str,
  horizon: int,
  primary: Candle,
  secondary: Candle,
  verified_at: datetime,
) -> dict:
  predictions = [
    p for p in ledger.records("predictions") if p["prediction_id"] == prediction_id
  ]
  if len(predictions) != 1:
    raise IntegrityError("Unknown or duplicate prediction")
  prediction = predictions[0]
  if prediction["status"] != "VALID" or not 1 <= horizon <= 24:
    raise ValueError("Invalid prediction/horizon")
  target = utc(prediction["forecast_origin"]) + horizon * HOUR
  verified_at = utc(verified_at)
  for candle in (primary, secondary):
    MarketDataValidator.validate([candle], verified_at, 1, fresh=False)
    if utc(candle.close_time) != target:
      raise ValueError("Observation does not match exact target")
  if (
    primary.provider != prediction["provider"]
    or primary.venue != prediction["venue"]
    or secondary.provider == primary.provider
  ):
    raise ValueError("Wrong primary or non-independent cross-check")
  a, b = Decimal(primary.close), Decimal(secondary.close)
  mean = (a + b) / 2
  spread = abs(a - b) / mean * 100
  identity = digest({"prediction": prediction_id, "horizon": horizon})
  sources = [c.to_dict() for c in (primary, secondary)]
  verification = {
    "schema_version": 1,
    "verification_id": identity,
    "prediction_id": prediction_id,
    "horizon": horizon,
    "target_timestamp": target.isoformat(),
    "verified_at": verified_at.isoformat(),
    "provider": primary.provider,
    "exact_price": primary.close,
    "source_timestamp": utc(primary.close_time).isoformat(),
    "sources": sources,
    "cross_source_mean": str(mean),
    "cross_source_median": str(mean),
    "source_spread_pct": str(spread),
    "status": "VERIFIED" if spread <= Decimal(1) else "SOURCE_DIVERGENCE",
  }
  with ledger.db:
    for source in sources:
      ledger.add(
        "observations", digest({"verification": identity, "source": source}), source
      )
    ledger.add("verifications", identity, verification)
  return verification
