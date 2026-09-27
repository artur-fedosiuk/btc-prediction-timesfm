"""Candidate same-quote providers. Adoption requires a successful CI probe."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from urllib.parse import urlencode

import requests

from .contracts import HOUR, Candle, MarketDataValidator, utc
from .pipeline import now


class ProviderUnavailable(RuntimeError):
  status = "PROVIDER_UNAVAILABLE"


class PublicMarket:
  def __init__(self, provider: str, *, transport=None, clock=now):
    if provider not in ("coinbase", "kraken"):
      raise ValueError("Unknown provider")
    self.http_status = None
    self.provider = provider
    self.transport = transport or self._get
    self.clock = clock

  def _get(self, url):
    try:
      response = requests.get(
        url, headers={"User-Agent": "BTC-forward-research/2"}, timeout=30
      )
      self.http_status = response.status_code
      response.raise_for_status()
    except requests.RequestException as exc:
      raise ProviderUnavailable(f"{type(exc).__name__}: {exc}") from exc
    return json.loads(response.text, parse_float=Decimal)

  def candles(self, start: datetime, end: datetime) -> list[Candle]:
    start, end = utc(start), utc(end)
    if end <= start or (end - start).total_seconds() % 3600:
      raise ValueError("Invalid interval")
    batches = []
    if self.provider == "coinbase":
      cursor = start
      while cursor < end:
        stop = min(cursor + 299 * HOUR, end)
        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?" + urlencode(
          {"granularity": 3600, "start": cursor.isoformat(), "end": stop.isoformat()}
        )
        rows = self.transport(url)
        if not isinstance(rows, list):
          raise ProviderUnavailable("Coinbase candle schema mismatch")
        batches.append((rows, utc(self.clock()), cursor, stop))
        cursor = stop
    else:
      url = "https://api.kraken.com/0/public/OHLC?" + urlencode(
        {"pair": "XBTUSD", "interval": 60, "since": int(start.timestamp())}
      )
      response = self.transport(url)
      if response.get("error"):
        raise ProviderUnavailable(str(response["error"]))
      result = response.get("result", {})
      if set(result) - {"last"} != {"XXBTZUSD"}:
        raise ProviderUnavailable("Unexpected Kraken pair")
      batches.append((result["XXBTZUSD"], utc(self.clock()), start, end))
    candles = []
    for rows, retrieved, left, right in batches:
      for row in rows:
        if len(row) != (6 if self.provider == "coinbase" else 8):
          raise ProviderUnavailable("Candle schema mismatch")
        opened = datetime.fromtimestamp(int(row[0]), timezone.utc)
        # End-exclusive range excludes Coinbase boundary duplicates and Kraken live tail.
        if not left <= opened < right:
          continue
        if self.provider == "coinbase":
          _, low, high, op, close, volume = row
        else:
          _, op, high, low, close, _vwap, volume, _count = row
        candles.append(
          Candle(
            self.provider,
            self.provider,
            "BTC/USD",
            "USD",
            opened,
            opened + HOUR,
            retrieved,
            retrieved,
            str(op),
            str(high),
            str(low),
            str(close),
            str(volume),
          )
        )
    # Coinbase's native order is newest-first. Normalize transport order, not missing data.
    candles.sort(key=lambda c: c.open_time)
    MarketDataValidator.validate(
      candles, self.clock(), int((end - start) / HOUR), fresh=False
    )
    return candles
