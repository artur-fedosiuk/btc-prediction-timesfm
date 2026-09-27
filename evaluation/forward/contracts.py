"""Central validation and temporal contracts for hourly BTC/USD experiments."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation

HOUR = timedelta(hours=1)
EXPERIMENT = "timesfm_h24_v2"
CHECKPOINT = "google/timesfm-3.0-pytorch"
REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"


class LookAheadBiasError(ValueError):
  pass


class DataError(ValueError):
  def __init__(self, status: str, message: str):
    self.status = status
    super().__init__(message)


def utc(value: str | datetime) -> datetime:
  result = (
    datetime.fromisoformat(value.replace("Z", "+00:00"))
    if isinstance(value, str)
    else value
  )
  if result.tzinfo is None or result.utcoffset() is None:
    raise ValueError("Naive timestamps are forbidden")
  return result.astimezone(timezone.utc)


def canonical(value) -> str:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
  return hashlib.sha256(canonical(value).encode()).hexdigest()


def positive(value) -> Decimal:
  try:
    result = Decimal(str(value))
  except (InvalidOperation, ValueError) as exc:
    raise DataError("INVALID_DATA", "Invalid numeric value") from exc
  if not result.is_finite() or result <= 0:
    raise DataError("INVALID_DATA", f"Invalid positive price: {value}")
  return result


@dataclass(frozen=True)
class Candle:
  provider: str
  venue: str
  symbol: str
  quote_currency: str
  open_time: datetime
  close_time: datetime
  available_at: datetime
  retrieved_at: datetime
  open: str
  high: str
  low: str
  close: str
  volume: str
  status: str = "CLOSED"

  def to_dict(self) -> dict:
    result = asdict(self)
    for name in ("open_time", "close_time", "available_at", "retrieved_at"):
      result[name] = utc(result[name]).isoformat()
    return result

  @classmethod
  def from_dict(cls, value: dict) -> Candle:
    value = dict(value)
    for name in ("open_time", "close_time", "available_at", "retrieved_at"):
      value[name] = utc(value[name])
    return cls(**value)


class MarketDataValidator:
  @staticmethod
  def validate(
    data: list[Candle], cutoff: datetime, required: int, *, fresh: bool = True
  ) -> datetime:
    cutoff = utc(cutoff)
    if len(data) != required or not data:
      raise DataError(
        "INSUFFICIENT_HISTORY", f"Expected {required} candles, got {len(data)}"
      )
    previous = None
    identity = (data[0].provider, data[0].venue)
    for candle in data:
      opened, closed, available, retrieved = map(
        utc,
        (candle.open_time, candle.close_time, candle.available_at, candle.retrieved_at),
      )
      if closed > cutoff or available > cutoff or retrieved > cutoff:
        raise LookAheadBiasError("Observation was not closed and available at cutoff")
      if available < closed or retrieved < available:
        raise DataError("INVALID_DATA", "Invalid availability order")
      if (
        candle.status != "CLOSED"
        or closed - opened != HOUR
        or opened.minute
        or opened.second
        or opened.microsecond
      ):
        raise DataError("INVALID_DATA", "Expected closed hourly candle on UTC grid")
      if previous is not None and opened != previous:
        raise DataError("INVALID_DATA", "Duplicate, missing or unordered candle")
      if (
        (candle.provider, candle.venue) != identity
        or not all(identity)
        or candle.symbol != "BTC/USD"
        or candle.quote_currency != "USD"
      ):
        raise DataError("INVALID_DATA", "Wrong market identity")
      o, h, l, c = map(positive, (candle.open, candle.high, candle.low, candle.close))
      volume = Decimal(candle.volume)
      if not volume.is_finite() or volume < 0 or not l <= min(o, c) <= max(o, c) <= h:
        raise DataError("INVALID_DATA", "Invalid OHLCV")
      previous = closed
    if fresh and previous != cutoff.replace(minute=0, second=0, microsecond=0):
      raise DataError("STALE_DATA", "Latest closed hourly candle is missing")
    return previous


class ForecastValidator:
  @staticmethod
  def validate(path, lower, upper, origin: datetime) -> list[dict]:
    origin = utc(origin)
    if any(len(values) != 24 for values in (path, lower, upper)):
      raise DataError("INVALID_FORECAST", "Expected 24 forecast points")
    points = []
    for step, (point, lo, hi) in enumerate(zip(path, lower, upper), 1):
      if (
        not all(math.isfinite(v) and v > 0 for v in (point, lo, hi))
        or not lo <= point <= hi
      ):
        raise DataError(
          "INVALID_FORECAST", "Nonfinite, nonpositive or unordered quantiles"
        )
      points.append(
        {
          "step": step,
          "target_timestamp": (origin + step * HOUR).isoformat(),
          "point_forecast": float(point),
          "q10": float(lo),
          "q90": float(hi),
        }
      )
    return points
