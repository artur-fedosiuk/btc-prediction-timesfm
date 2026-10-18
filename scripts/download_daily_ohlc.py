"""Download daily OHLC candles for BTC-USD from Coinbase Exchange.

Source: GET https://api.exchange.coinbase.com/products/{product}/candles
  granularity=86400 (daily)
  Response row order: [timestamp, low, high, open, close, volume]  (newest-first)
  Max 300 candles per request (Coinbase docs, verified 2026-10-12).

Run as:
    python -m scripts.download_daily_ohlc [--years N] [--out PATH] [--product ID]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.parse import urlencode

import requests as _requests

from scripts.ohlc_io import Candle, save_csv, validate_series

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GRANULARITY: int = 86400  # daily candles, seconds
_CANDLE_LIMIT: int = 300  # Coinbase max candles per request
_WINDOW_DAYS: int = 290  # request window: 10 below limit (safety margin)
_MAX_ATTEMPTS: int = 5
_MIN_PAUSE_S: float = 0.2  # politeness delay between windows
_USER_AGENT: str = "btc-oracle-research/1 (non-commercial, research only)"
_BASE_URL: str = "https://api.exchange.coinbase.com"

_UTC = timezone.utc

# Status codes that trigger a retry (429 + all 5xx)
_RETRYABLE: frozenset[int] = frozenset(range(500, 600)) | {429}

# ---------------------------------------------------------------------------
# HTTP client protocol and real implementation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawResponse:
  """Minimal parsed HTTP response; returned by any HttpClient implementation."""

  status_code: int
  data: Any  # parsed JSON body; None when not parseable
  retry_after: float | None = None  # seconds from Retry-After header


class HttpClient(Protocol):
  """Minimal injectable HTTP client. Implement with a fake in tests."""

  def get(self, url: str) -> RawResponse:
    """Perform a GET and return a RawResponse (never raise)."""
    ...


class _RequestsClient:
  """Real HTTP client backed by the `requests` library."""

  def get(self, url: str) -> RawResponse:
    retry_after: float | None = None
    try:
      resp = _requests.get(
        url,
        headers={"User-Agent": _USER_AGENT},
        timeout=10,
      )
    except _requests.Timeout:
      _LOG.warning("Request timed out: %s", url)
      # Treat timeout as retriable via a synthetic 503
      return RawResponse(status_code=503, data=None)
    except _requests.RequestException as exc:
      _LOG.warning("Request failed (%s): %s", type(exc).__name__, exc)
      return RawResponse(status_code=503, data=None)

    if resp.status_code == 429:
      try:
        retry_after = float(resp.headers.get("Retry-After", "0") or "0")
      except (ValueError, TypeError):
        retry_after = None

    try:
      data: Any = resp.json()
    except Exception:
      data = None

    return RawResponse(
      status_code=resp.status_code,
      data=data,
      retry_after=retry_after,
    )


# ---------------------------------------------------------------------------
# URL builder
# ---------------------------------------------------------------------------


def _build_url(product: str, start: datetime, end: datetime) -> str:
  """Build the Coinbase candles endpoint URL for the window [start, end)."""
  params = urlencode({
    "granularity": _GRANULARITY,
    "start": start.isoformat(),
    "end": end.isoformat(),
  })
  return f"{_BASE_URL}/products/{product}/candles?{params}"


# ---------------------------------------------------------------------------
# Candle parser (client-side filtering + deduplication)
# ---------------------------------------------------------------------------


def _parse_and_filter(
  data: Any,
  window_start: datetime,
  window_end: datetime,
) -> list[Candle]:
  """Parse Coinbase candle rows; keep only those strictly inside [start, end).

  The Coinbase docs note that returned candles may precede `start`, so we
  filter client-side.  Duplicate timestamps (same date returned twice) are
  silently de-duplicated, keeping the first occurrence.

  Row format (documented, verified in providers.py L78):
    [timestamp, low, high, open, close, volume]

  Raises ValueError if any timestamp is not aligned to 00:00 UTC.
  """
  if not isinstance(data, list):
    raise ValueError(
      f"Expected a JSON list from Coinbase candles endpoint, "
      f"got {type(data).__name__!r}"
    )

  candles: list[Candle] = []
  seen_dates: set[date] = set()

  for row in data:
    if len(row) != 6:
      raise ValueError(
        f"Expected 6 fields in candle row, got {len(row)}: {row!r}"
      )
    ts_raw, low_raw, high_raw, open_raw, close_raw, vol_raw = row
    ts = int(ts_raw)
    opened = datetime.fromtimestamp(ts, tz=_UTC)

    # Filter: keep only candles in [window_start, window_end)
    if not (window_start <= opened < window_end):
      _LOG.debug("Discarding out-of-range candle at %s", opened.date())
      continue

    # Daily candles must be aligned to 00:00 UTC
    if ts % _GRANULARITY != 0:
      raise ValueError(
        f"Candle timestamp {ts} is not aligned to 00:00 UTC "
        f"(ts % {_GRANULARITY} = {ts % _GRANULARITY})"
      )

    opened_date = opened.date()
    if opened_date in seen_dates:
      _LOG.debug("De-duplicating repeated candle at %s", opened_date)
      continue
    seen_dates.add(opened_date)

    candles.append(
      Candle(
        date_utc=opened_date,
        open=float(open_raw),
        high=float(high_raw),
        low=float(low_raw),
        close=float(close_raw),
        volume=float(vol_raw),
      )
    )

  return candles


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_window(
  client: HttpClient,
  product: str,
  start: datetime,
  end: datetime,
  *,
  sleep_fn: Callable[[float], None] = time.sleep,
) -> list[Candle]:
  """Fetch one time window of candles with exponential-backoff retry.

  Retries on HTTP 429, 5xx, and network timeouts (presented as synthetic 503).
  Raises immediately on any other 4xx status.
  Honors the Retry-After header for 429 responses.
  """
  url = _build_url(product, start, end)
  delay = 1.0

  for attempt in range(1, _MAX_ATTEMPTS + 1):
    resp = client.get(url)

    if resp.status_code == 200:
      return _parse_and_filter(resp.data, start, end)

    # Non-retryable 4xx (e.g. 400, 401, 403, 404) → raise immediately
    if 400 <= resp.status_code < 500 and resp.status_code not in _RETRYABLE:
      raise RuntimeError(
        f"Non-retryable HTTP {resp.status_code} from Coinbase "
        f"(url={url!r})"
      )

    if resp.status_code not in _RETRYABLE:
      raise RuntimeError(
        f"Unexpected HTTP {resp.status_code} from Coinbase (url={url!r})"
      )

    if attempt == _MAX_ATTEMPTS:
      raise RuntimeError(
        f"HTTP {resp.status_code} persisted after {_MAX_ATTEMPTS} attempts "
        f"(url={url!r})"
      )

    # Use Retry-After for 429; otherwise use exponential backoff
    wait = (resp.retry_after or delay) if resp.status_code == 429 else delay
    _LOG.warning(
      "HTTP %d on attempt %d/%d; sleeping %.1f s before retry",
      resp.status_code,
      attempt,
      _MAX_ATTEMPTS,
      wait,
    )
    sleep_fn(wait)
    delay *= 2

  raise RuntimeError("Retry loop exhausted without success")  # unreachable


def download(
  client: HttpClient,
  product: str,
  years: int,
  end_date: date,
  sleep_fn: Callable[[float], None] = time.sleep,
) -> list[Candle]:
  """Download `years` of daily candles ending (exclusive) at `end_date`.

  Paginates backwards in windows of _WINDOW_DAYS days.
  De-duplicates by date across windows (keeps first occurrence).
  Returns candles sorted ascending by date.
  """
  end_dt = datetime(end_date.year, end_date.month, end_date.day, tzinfo=_UTC)
  start_dt = end_dt - timedelta(days=years * 365)

  seen: set[date] = set()
  all_candles: list[Candle] = []

  window_end = end_dt
  while window_end > start_dt:
    window_start = max(start_dt, window_end - timedelta(days=_WINDOW_DAYS))
    _LOG.info(
      "Fetching window [%s, %s)",
      window_start.date(),
      window_end.date(),
    )
    batch = fetch_window(
      client, product, window_start, window_end, sleep_fn=sleep_fn
    )
    for c in batch:
      if c.date_utc not in seen:
        seen.add(c.date_utc)
        all_candles.append(c)
    window_end = window_start
    sleep_fn(_MIN_PAUSE_S)

  all_candles.sort(key=lambda c: c.date_utc)
  return all_candles


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _sha256_of_file(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_meta(candles: list[Candle], csv_path: Path, product: str) -> None:
  """Write a JSON sidecar with provenance metadata next to the CSV."""
  meta = {
    "source": "Coinbase Exchange REST API",
    "product": product,
    "granularity": _GRANULARITY,
    "fetched_at_utc": datetime.now(_UTC).isoformat(),
    "rows": len(candles),
    "first_date": str(candles[0].date_utc),
    "last_date": str(candles[-1].date_utc),
    "sha256": _sha256_of_file(csv_path),
  }
  meta_path = csv_path.parent / (csv_path.stem + ".meta.json")
  meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
  _LOG.info("Metadata written to %s", meta_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
  )
  parser = argparse.ArgumentParser(
    description="Download daily BTC-USD OHLC candles from Coinbase Exchange."
  )
  parser.add_argument(
    "--years",
    type=int,
    default=5,
    help="Years of history to download (default: 5)",
  )
  parser.add_argument(
    "--out",
    default="research/data/btc_usd_1d.csv",
    help="Output CSV path (default: research/data/btc_usd_1d.csv)",
  )
  parser.add_argument(
    "--product",
    default="BTC-USD",
    help="Coinbase product ID (default: BTC-USD)",
  )
  args = parser.parse_args()

  today_utc = datetime.now(_UTC).date()
  # Exclusive end date: today at 00:00 UTC (excludes today's incomplete candle,
  # so the last downloaded candle is yesterday, the last complete UTC day).
  end_date = today_utc

  _LOG.info(
    "Downloading %d years of %s daily candles ending %s (exclusive)",
    args.years,
    args.product,
    end_date,
  )

  client = _RequestsClient()
  candles = download(client, args.product, args.years, end_date)

  try:
    validate_series(candles)
  except ValueError as exc:
    print(f"ERROR: series validation failed — {exc}", file=sys.stderr)
    sys.exit(1)

  out_path = Path(args.out)
  save_csv(candles, out_path)
  _write_meta(candles, out_path, args.product)

  # CLI summary (only print() in this file)
  sha = _sha256_of_file(out_path)
  print(f"Rows      : {len(candles)}")
  print(f"First date: {candles[0].date_utc}")
  print(f"Last date : {candles[-1].date_utc}")
  print(f"Output    : {out_path}")
  print(f"SHA-256   : {sha}")


if __name__ == "__main__":
  main()
