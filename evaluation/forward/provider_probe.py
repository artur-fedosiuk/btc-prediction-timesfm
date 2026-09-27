"""Read-only local/CI provider reachability and schema evidence."""

import json
import os
from pathlib import Path

from .contracts import HOUR, MarketDataValidator
from .pipeline import now
from .providers import PublicMarket
from .storage import atomic_write


def probe(output: Path) -> dict:
  end = now().replace(minute=0, second=0, microsecond=0)
  result = {
    "checked_at": now().isoformat(),
    "github_run_id": os.getenv("GITHUB_RUN_ID"),
    "github_sha": os.getenv("GITHUB_SHA"),
    "environment": "CI" if os.getenv("GITHUB_ACTIONS") else "LOCAL",
    "providers": [],
  }
  for name in ("coinbase", "kraken"):
    market = PublicMarket(name)
    try:
      candles = market.candles(end - 4 * HOUR, end)
      MarketDataValidator.validate(candles, now(), 4)
      item = {
        "provider": name,
        "status": "AVAILABLE",
        "count": len(candles),
        "http_status": market.http_status,
        "validation_status": "PASS",
        "freshness_seconds": (now() - candles[-1].close_time).total_seconds(),
        "latest": candles[-1].to_dict(),
      }
    except Exception as exc:  # noqa: BLE001 -- Persist boundary failure and return nonzero.
      item = {
        "provider": name,
        "status": "PROVIDER_UNAVAILABLE",
        "http_status": market.http_status,
        "validation_status": "FAIL",
        "error": f"{type(exc).__name__}: {exc}",
      }
    result["providers"].append(item)
  atomic_write(output, json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))
  return result


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--output", type=Path, default=Path("reports/provider-checks.json")
  )
  args = parser.parse_args()
  result = probe(args.output)
  raise SystemExit(
    0 if all(p["status"] == "AVAILABLE" for p in result["providers"]) else 1
  )
