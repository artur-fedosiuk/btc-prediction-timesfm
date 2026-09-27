from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from evaluation.forward.contracts import (
  Candle,
  ForecastValidator,
  MarketDataValidator,
  utc,
)

T = datetime(2026, 9, 27, 12, tzinfo=timezone.utc)


def candles(count=4, price="100"):
  return [
    Candle(
      "coinbase",
      "coinbase",
      "BTC/USD",
      "USD",
      T - timedelta(hours=count - i),
      T - timedelta(hours=count - i - 1),
      T,
      T,
      price,
      price,
      price,
      price,
      "1",
    )
    for i in range(count)
  ]


def test_grid_and_horizons():
  assert MarketDataValidator.validate(candles(), T, 4) == T
  points = ForecastValidator.validate([110.0] * 24, [90.0] * 24, [120.0] * 24, T)
  assert [points[i - 1]["target_timestamp"] for i in (1, 4, 8, 12, 24)] == [
    (T + timedelta(hours=i)).isoformat() for i in (1, 4, 8, 12, 24)
  ]


@pytest.mark.parametrize("price", ["NaN", "Infinity", "0", "-1"])
def test_invalid_prices(price):
  with pytest.raises(ValueError):
    MarketDataValidator.validate(candles(price=price), T, 4)


@pytest.mark.parametrize(
  "mutation",
  ["duplicate", "reverse", "missing", "stale", "open", "available", "symbol", "quote"],
)
def test_invalid_grid(mutation):
  data = candles()
  cutoff = T
  if mutation == "duplicate":
    data[1] = data[0]
  if mutation == "reverse":
    data.reverse()
  if mutation == "missing":
    data.pop(1)
  if mutation == "stale":
    cutoff += timedelta(hours=2)
  if mutation == "open":
    data[-1] = replace(data[-1], close_time=T + timedelta(hours=1))
  if mutation == "available":
    data[-1] = replace(data[-1], available_at=T + timedelta(seconds=1))
  if mutation == "symbol":
    data[-1] = replace(data[-1], symbol="ETH/USD")
  if mutation == "quote":
    data[-1] = replace(data[-1], quote_currency="USDT")
  with pytest.raises(ValueError):
    MarketDataValidator.validate(data, cutoff, 4)


@pytest.mark.parametrize(
  "timestamp",
  ["2026-03-29T03:00:00+02:00", "2026-03-29T02:00:00+01:00", "2026-03-29T01:00:00Z"],
)
def test_dst_start(timestamp):
  assert utc(timestamp) == datetime(2026, 3, 29, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize(
  "timestamp",
  ["2026-10-25T02:00:00+02:00", "2026-10-25T01:00:00+01:00", "2026-10-25T00:00:00Z"],
)
def test_dst_end(timestamp):
  assert utc(timestamp) == datetime(2026, 10, 25, 0, tzinfo=timezone.utc)


def test_naive_rejected():
  with pytest.raises(ValueError):
    utc("2026-09-27T12:00:00")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 0])
def test_invalid_forecast(value):
  with pytest.raises(ValueError):
    ForecastValidator.validate([value] * 24, [90.0] * 24, [120.0] * 24, T)


def test_quantile_order():
  with pytest.raises(ValueError):
    ForecastValidator.validate([100.0] * 24, [101.0] * 24, [120.0] * 24, T)
