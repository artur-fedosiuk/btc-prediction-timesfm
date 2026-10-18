"""Tests for scripts/download_daily_ohlc.py and scripts/ohlc_io.py.

NO network access: an autouse fixture makes socket.socket raise immediately
for every test in this module, proving the implementation uses only fakes.
"""
from __future__ import annotations

import socket
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from scripts.download_daily_ohlc import (
  RawResponse,
  _WINDOW_DAYS,
  download,
  fetch_window,
)
from scripts.ohlc_io import Candle, load_csv, save_csv, validate_series

_UTC = timezone.utc

# ---------------------------------------------------------------------------
# Autouse fixture: block all real network access
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
  """Any attempt to open a real socket raises OSError."""

  def _raise(*args: Any, **kwargs: Any) -> None:
    raise OSError("Network access is forbidden in unit tests")

  monkeypatch.setattr(socket, "socket", _raise)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_api_row(d: date, price: float = 10_000.0) -> list[Any]:
  """Return a synthetic Coinbase candle row: [ts, low, high, open, close, vol]."""
  ts = int(datetime(d.year, d.month, d.day, tzinfo=_UTC).timestamp())
  return [ts, price - 50.0, price + 50.0, price - 10.0, price + 10.0, 1.0]


@dataclass
class FakeClient:
  """Returns pre-loaded RawResponse objects in sequence."""

  responses: list[RawResponse]
  call_count: int = field(default=0, init=False, repr=False)

  def get(self, url: str) -> RawResponse:  # noqa: ARG002
    resp = self.responses[self.call_count]
    self.call_count += 1
    return resp


@dataclass
class SyntheticClient:
  """Generates candle data for any requested [start, end) window.

  Produces one candle per calendar day that is in `available`.
  Rows are returned newest-first (matching Coinbase's native order).
  """

  available: set[date]

  @classmethod
  def for_range(cls, start: date, end_exclusive: date) -> "SyntheticClient":
    days = {
      start + timedelta(days=i)
      for i in range((end_exclusive - start).days)
    }
    return cls(available=days)

  def get(self, url: str) -> RawResponse:
    params = parse_qs(urlparse(url).query)
    ws = datetime.fromisoformat(params["start"][0]).replace(tzinfo=_UTC)
    we = datetime.fromisoformat(params["end"][0]).replace(tzinfo=_UTC)
    rows: list[list[Any]] = []
    d = ws.date()
    while d < we.date():
      if d in self.available:
        rows.append(_make_api_row(d))
      d += timedelta(days=1)
    rows.reverse()  # newest-first, as the real API sends
    return RawResponse(status_code=200, data=rows)


def _simple_candle(d: date, price: float = 100.0) -> Candle:
  """Helper to build a valid Candle for validate_series tests."""
  return Candle(
    date_utc=d,
    open=price,
    high=price + 1.0,
    low=price - 1.0,
    close=price,
    volume=1.0,
  )


# ---------------------------------------------------------------------------
# Test 1 – pagination over a multi-window series
# ---------------------------------------------------------------------------


def test_pagination_no_duplicates_no_gaps() -> None:
  """download() over 1095 days (3 years, 4 windows) returns exactly 1095 candles
  with no duplicates and no gaps.

  Windows: [end-290, end), [end-580, end-290), [end-870, end-580),
           [end-1095, end-870)  (last window = 225 days, not a multiple).
  """
  end_date = date(2023, 12, 31)
  years = 3
  expected_days = years * 365  # 1095
  start_date = end_date - timedelta(days=expected_days)

  client = SyntheticClient.for_range(start_date, end_date)
  sleep_calls: list[float] = []

  candles = download(
    client, "BTC-USD", years, end_date, sleep_fn=sleep_calls.append
  )

  dates = [c.date_utc for c in candles]
  assert len(candles) == expected_days, (
    f"Expected {expected_days} candles, got {len(candles)}"
  )
  assert len(set(dates)) == expected_days, "Duplicate dates in result"
  assert dates == sorted(dates), "Result not sorted ascending"
  for i in range(1, len(dates)):
    assert dates[i] - dates[i - 1] == timedelta(days=1), (
      f"Gap between {dates[i - 1]} and {dates[i]}"
    )


# ---------------------------------------------------------------------------
# Test 2 – out-of-range candles and repeated boundary candle
# ---------------------------------------------------------------------------


def test_out_of_range_and_repeated_boundary_candle() -> None:
  """Client returns: one candle before start, two identical candles inside the
  window, one candle at end (exclusive).  Result must contain exactly the two
  unique in-range dates, with no duplicates.
  """
  d_before = date(2023, 1, 14)  # before window start → filtered
  d1 = date(2023, 1, 15)  # in [start, end)
  d2 = date(2023, 1, 16)  # in [start, end)
  d_end = date(2023, 1, 17)  # == window end (exclusive) → filtered

  client = FakeClient(
    responses=[
      RawResponse(
        status_code=200,
        data=[
          _make_api_row(d_before),  # out-of-range: before start
          _make_api_row(d2),
          _make_api_row(d1),
          _make_api_row(d1),  # duplicate of d1 → de-duplicated
          _make_api_row(d_end),  # out-of-range: at exclusive end
        ],
      )
    ]
  )
  start_dt = datetime(2023, 1, 15, tzinfo=_UTC)
  end_dt = datetime(2023, 1, 17, tzinfo=_UTC)

  candles = fetch_window(client, "BTC-USD", start_dt, end_dt)
  dates = [c.date_utc for c in candles]

  assert len(candles) == 2, f"Expected 2 candles, got {len(candles)}: {dates}"
  assert d_before not in dates, "Pre-start candle leaked through"
  assert d_end not in dates, "End-boundary candle leaked through"
  assert d1 in dates and d2 in dates, "In-range candle missing"
  assert dates.count(d1) == 1, f"d1 appears {dates.count(d1)} times (expected 1)"


# ---------------------------------------------------------------------------
# Test 3 – series length not a multiple of window size
# ---------------------------------------------------------------------------


def test_non_multiple_window_size() -> None:
  """1 year = 365 days; 365 % _WINDOW_DAYS != 0, forcing unequal window sizes.

  This ensures the last pagination window (a remainder) is handled correctly.
  """
  assert 365 % _WINDOW_DAYS != 0, "Test assumption invalid for current _WINDOW_DAYS"
  end_date = date(2023, 1, 1)
  years = 1
  expected_days = years * 365
  start_date = end_date - timedelta(days=expected_days)

  client = SyntheticClient.for_range(start_date, end_date)
  candles = download(
    client, "BTC-USD", years, end_date, sleep_fn=lambda _: None
  )

  assert len(candles) == expected_days, (
    f"Expected {expected_days}, got {len(candles)}"
  )
  dates = [c.date_utc for c in candles]
  assert dates == sorted(set(dates))
  for i in range(1, len(dates)):
    assert dates[i] - dates[i - 1] == timedelta(days=1), (
      f"Gap between {dates[i - 1]} and {dates[i]}"
    )


# ---------------------------------------------------------------------------
# Test 4 – field order parsing
# ---------------------------------------------------------------------------


def test_field_order_low_high_open_close() -> None:
  """Coinbase row [ts, LOW, HIGH, open, close, vol] must map to the right fields.

  This test would fail if low and high were swapped in the parser.
  """
  d = date(2023, 3, 15)
  ts = int(datetime(2023, 3, 15, tzinfo=_UTC).timestamp())
  # Documented order: [timestamp, price_low, price_high, price_open, price_close, volume]
  #                                   19000     21000      20000       20500
  row: list[Any] = [ts, 19_000.0, 21_000.0, 20_000.0, 20_500.0, 123.456]

  client = FakeClient(responses=[RawResponse(status_code=200, data=[row])])
  start_dt = datetime(2023, 3, 15, tzinfo=_UTC)
  end_dt = datetime(2023, 3, 16, tzinfo=_UTC)

  candles = fetch_window(client, "BTC-USD", start_dt, end_dt)

  assert len(candles) == 1
  c = candles[0]
  assert c.date_utc == d
  assert c.low == 19_000.0, f"low={c.low!r} (expected 19000; swapped with high?)"
  assert c.high == 21_000.0, f"high={c.high!r} (expected 21000; swapped with low?)"
  assert c.open == 20_000.0, f"open={c.open!r} (expected 20000)"
  assert c.close == 20_500.0, f"close={c.close!r} (expected 20500)"
  assert c.volume == 123.456


# ---------------------------------------------------------------------------
# Test 5 – retry logic
# ---------------------------------------------------------------------------


def test_retry_429_twice_then_200_succeeds() -> None:
  """fetch_window: 429 on attempts 1 and 2, 200 on attempt 3 → success.
  sleep_fn is called with increasing delays (exponential backoff).
  """
  d = date(2023, 3, 15)
  ts = int(datetime(2023, 3, 15, tzinfo=_UTC).timestamp())
  row: list[Any] = [ts, 9_000.0, 11_000.0, 10_000.0, 10_100.0, 1.0]

  client = FakeClient(
    responses=[
      RawResponse(status_code=429, data=None),
      RawResponse(status_code=429, data=None),
      RawResponse(status_code=200, data=[row]),
    ]
  )
  sleep_calls: list[float] = []
  start_dt = datetime(2023, 3, 15, tzinfo=_UTC)
  end_dt = datetime(2023, 3, 16, tzinfo=_UTC)

  candles = fetch_window(
    client, "BTC-USD", start_dt, end_dt, sleep_fn=sleep_calls.append
  )

  assert len(candles) == 1, "Expected 1 candle after retry"
  assert candles[0].date_utc == d
  assert client.call_count == 3, f"Expected 3 HTTP calls, got {client.call_count}"
  assert len(sleep_calls) == 2, f"Expected 2 sleep calls, got {sleep_calls}"
  assert sleep_calls[0] < sleep_calls[1], (
    f"Delays not increasing: {sleep_calls} (expected exponential backoff)"
  )


def test_retry_400_raises_immediately() -> None:
  """fetch_window: HTTP 400 raises RuntimeError on the first attempt; no retry."""
  client = FakeClient(responses=[RawResponse(status_code=400, data=None)])
  sleep_calls: list[float] = []
  start_dt = datetime(2023, 3, 15, tzinfo=_UTC)
  end_dt = datetime(2023, 3, 16, tzinfo=_UTC)

  with pytest.raises(RuntimeError, match="400"):
    fetch_window(
      client, "BTC-USD", start_dt, end_dt, sleep_fn=sleep_calls.append
    )

  assert client.call_count == 1, "400 must not be retried"
  assert sleep_calls == [], "sleep must not be called for a 400 error"


# ---------------------------------------------------------------------------
# Test 6 – validate_series rejections
# ---------------------------------------------------------------------------


def test_validate_rejects_duplicate_date() -> None:
  d = date(2023, 1, 1)
  with pytest.raises(ValueError, match="[Dd]uplicate"):
    validate_series([_simple_candle(d), _simple_candle(d)])


def test_validate_rejects_missing_day() -> None:
  candles = [
    _simple_candle(date(2023, 1, 1)),
    _simple_candle(date(2023, 1, 3)),  # 2023-01-02 is missing
  ]
  with pytest.raises(ValueError, match="2023-01-02"):
    validate_series(candles)


def test_validate_rejects_high_lt_close() -> None:
  """high must be >= max(open, close, low)."""
  c = Candle(
    date_utc=date(2023, 1, 1),
    open=100.0,
    high=99.0,  # < close=105 → invalid
    low=98.0,
    close=105.0,
    volume=1.0,
  )
  with pytest.raises(ValueError, match="high"):
    validate_series([c])


def test_validate_rejects_negative_volume() -> None:
  c = Candle(
    date_utc=date(2023, 1, 1),
    open=100.0,
    high=101.0,
    low=99.0,
    close=100.5,
    volume=-1.0,
  )
  with pytest.raises(ValueError, match="[Nn]egative"):
    validate_series([c])


def test_parse_rejects_misaligned_timestamp() -> None:
  """A timestamp not at 00:00 UTC must raise ValueError during parsing."""
  ts_aligned = int(datetime(2023, 1, 1, tzinfo=_UTC).timestamp())
  ts_misaligned = ts_aligned + 3_600  # +1 hour → not 00:00 UTC
  row: list[Any] = [ts_misaligned, 99.0, 101.0, 100.0, 100.5, 1.0]

  client = FakeClient(responses=[RawResponse(status_code=200, data=[row])])
  start_dt = datetime(2023, 1, 1, tzinfo=_UTC)
  end_dt = datetime(2023, 1, 2, tzinfo=_UTC)

  # The misaligned candle is inside [start, end) — filter does not block it,
  # but the alignment check must raise.
  with pytest.raises(ValueError, match="aligned"):
    fetch_window(client, "BTC-USD", start_dt, end_dt)


# ---------------------------------------------------------------------------
# Test 7 – CSV round-trip
# ---------------------------------------------------------------------------


def test_csv_roundtrip(tmp_path: Path) -> None:
  """save_csv then load_csv returns bit-for-bit equal Candle objects."""
  # Use prices that stress float repr (not exact in decimal but exact in IEEE 754)
  candles = [
    Candle(
      date_utc=date(2023, 1, 1),
      open=29_000.1,
      high=29_500.9,
      low=28_800.3,
      close=29_100.7,
      volume=1234.5678,
    ),
    Candle(
      date_utc=date(2023, 1, 2),
      open=29_100.7,
      high=30_000.0,
      low=29_000.0,
      close=29_750.25,
      volume=999.0,
    ),
    Candle(
      date_utc=date(2023, 1, 3),
      open=29_750.25,
      high=31_000.5,
      low=29_500.0,
      close=30_888.88,
      volume=0.001,
    ),
  ]
  csv_path = tmp_path / "roundtrip.csv"
  save_csv(candles, csv_path)
  loaded = load_csv(csv_path)

  assert loaded == candles, (
    f"Round-trip mismatch.\nOriginal: {candles}\nLoaded:   {loaded}"
  )
