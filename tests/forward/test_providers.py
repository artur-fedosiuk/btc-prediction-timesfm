from datetime import timedelta

import pytest

from evaluation.forward.contracts import MarketDataValidator
from evaluation.forward.providers import ProviderUnavailable, PublicMarket

from .test_contracts import T


def test_coinbase_native_order_and_live_tail():
  rows = [
    [int((T + timedelta(hours=i)).timestamp()), 99, 101, 100, 100, 1]
    for i in (0, -1, -2)
  ]
  market = PublicMarket("coinbase", transport=lambda url: rows, clock=lambda: T)
  data = market.candles(T - timedelta(hours=2), T)
  assert len(data) == 2
  assert MarketDataValidator.validate(data, T, 2) == T


def test_provider_duplicates_rejected():
  row = [int((T - timedelta(hours=1)).timestamp()), 99, 101, 100, 100, 1]
  market = PublicMarket("coinbase", transport=lambda url: [row, row], clock=lambda: T)
  with pytest.raises(ValueError):
    market.candles(T - timedelta(hours=2), T)


def test_kraken_exact_values_and_open_tail():
  def get(url):
    return {
      "error": [],
      "result": {
        "last": 0,
        "XXBTZUSD": [
          [
            int((T + timedelta(hours=i)).timestamp()),
            "100",
            "101",
            "99",
            "100.123456789",
            "100",
            "1",
            4,
          ]
          for i in (-1, 0)
        ],
      },
    }

  data = PublicMarket("kraken", transport=get, clock=lambda: T).candles(
    T - timedelta(hours=1), T
  )
  assert len(data) == 1
  assert data[0].close == "100.123456789"


def test_wrong_kraken_pair_rejected():
  market = PublicMarket(
    "kraken", transport=lambda url: {"result": {"XETHZUSD": []}}, clock=lambda: T
  )
  with pytest.raises(ProviderUnavailable):
    market.candles(T - timedelta(hours=1), T)


@pytest.mark.parametrize("step", [0, 1, -1])
def test_constant_uptrend_downtrend(step):
  from dataclasses import replace

  from .test_contracts import candles

  data = [
    replace(
      c,
      open=str(100 + step * i),
      high=str(100 + step * i),
      low=str(100 + step * i),
      close=str(100 + step * i),
    )
    for i, c in enumerate(candles())
  ]
  assert MarketDataValidator.validate(data, T, 4) == T
