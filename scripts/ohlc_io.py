"""Shared OHLC data types and IO utilities for the oracle research scripts.

Used by both scripts/download_daily_ohlc.py and scripts/oracle_breakeven.py.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candle:
  """One daily OHLC candle (UTC open date, prices in quote currency)."""

  date_utc: date
  open: float
  high: float
  low: float
  close: float
  volume: float


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_series(candles: list[Candle]) -> None:
  """Raise ValueError describing every integrity problem found.

  Checks (in this order):
  - empty input
  - NaN / inf in any numeric field
  - non-positive price (open, high, low, close)
  - negative volume
  - high < max(open, close, low)
  - low > min(open, close, high)
  - duplicate dates
  - dates not strictly ascending
  - missing calendar days (all missing dates listed in the message)
  """
  if not candles:
    raise ValueError("Empty candle series.")

  for c in candles:
    _check_candle(c)

  dates = [c.date_utc for c in candles]
  for i in range(1, len(dates)):
    if dates[i] == dates[i - 1]:
      raise ValueError(f"Duplicate date: {dates[i]}")
    if dates[i] < dates[i - 1]:
      raise ValueError(
        f"Dates not strictly ascending: {dates[i - 1]} then {dates[i]}"
      )

  missing: list[str] = []
  for i in range(1, len(dates)):
    expected = dates[i - 1] + timedelta(days=1)
    while expected < dates[i]:
      missing.append(str(expected))
      expected += timedelta(days=1)
  if missing:
    raise ValueError(f"Missing calendar days: {', '.join(missing)}")


def _check_candle(c: Candle) -> None:
  """Raise ValueError if a single candle is internally inconsistent."""
  for name, val in (
    ("open", c.open),
    ("high", c.high),
    ("low", c.low),
    ("close", c.close),
  ):
    if not math.isfinite(val):
      raise ValueError(f"Non-finite {name}={val} at {c.date_utc}")
    if val <= 0:
      raise ValueError(f"Non-positive {name}={val} at {c.date_utc}")

  if not math.isfinite(c.volume):
    raise ValueError(f"Non-finite volume={c.volume} at {c.date_utc}")
  if c.volume < 0:
    raise ValueError(f"Negative volume={c.volume} at {c.date_utc}")

  max_price = max(c.open, c.close, c.low)
  if c.high < max_price:
    raise ValueError(
      f"high={c.high} < max(open={c.open}, close={c.close}, low={c.low})"
      f"={max_price} at {c.date_utc}"
    )
  min_price = min(c.open, c.close, c.high)
  if c.low > min_price:
    raise ValueError(
      f"low={c.low} > min(open={c.open}, close={c.close}, high={c.high})"
      f"={min_price} at {c.date_utc}"
    )


# ---------------------------------------------------------------------------
# CSV IO
# ---------------------------------------------------------------------------

_CSV_HEADER: tuple[str, ...] = ("date_utc", "open", "high", "low", "close", "volume")
_DATE_FMT: str = "%Y-%m-%d"


def save_csv(candles: list[Candle], path: Path | str) -> None:
  """Write candles to CSV; floats written with repr() for exact round-trip.

  Header: date_utc,open,high,low,close,volume
  Dates:  YYYY-MM-DD
  """
  dest = Path(path)
  dest.parent.mkdir(parents=True, exist_ok=True)
  with dest.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)
    writer.writerow(_CSV_HEADER)
    for c in candles:
      writer.writerow([
        c.date_utc.strftime(_DATE_FMT),
        repr(c.open),
        repr(c.high),
        repr(c.low),
        repr(c.close),
        repr(c.volume),
      ])


def load_csv(path: Path | str) -> list[Candle]:
  """Load candles from CSV and call validate_series before returning."""
  src = Path(path)
  candles: list[Candle] = []
  with src.open(newline="", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
      candles.append(
        Candle(
          date_utc=date.fromisoformat(row["date_utc"]),
          open=float(row["open"]),
          high=float(row["high"]),
          low=float(row["low"]),
          close=float(row["close"]),
          volume=float(row["volume"]),
        )
      )
  validate_series(candles)
  return candles
