"""Separate commands for prediction, verification, export and provider checks."""

import argparse
import json
from decimal import Decimal
from pathlib import Path

from .analytics import write_report
from .contracts import HOUR, MarketDataValidator, utc
from .model import TimesFM
from .pipeline import generate, now, record_event, verify
from .providers import PublicMarket
from .storage import Ledger


def require_ci_evidence(path):
  evidence = json.loads(Path(path).read_text())
  if evidence.get("environment") != "CI" or not evidence.get("github_run_id"):
    raise RuntimeError(
      "CI provider adoption evidence is required before live generation"
    )
  if {p["provider"] for p in evidence["providers"] if p["status"] == "AVAILABLE"} != {
    "coinbase",
    "kraken",
  }:
    raise RuntimeError("Both same-quote providers must pass CI checks")


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument("command", choices=["generate", "verify", "report"])
  parser.add_argument("--ledger", type=Path, default=Path("data/forward.sqlite"))
  parser.add_argument("--ci-evidence", default="reports/provider-checks-ci.json")
  parser.add_argument("--output-dir", type=Path, default=Path("."))
  args = parser.parse_args(argv)
  ledger = Ledger(args.ledger)
  failures = []
  try:
    if args.command == "generate":
      require_ci_evidence(args.ci_evidence)
      end = now().replace(minute=0, second=0, microsecond=0)
      if any(utc(p["forecast_origin"]) == end for p in ledger.records("predictions")):
        record_event(
          ledger,
          "ALREADY_EXISTS",
          "Generation rerun skipped explicitly",
          operation="generate",
        )
      else:
        primary = PublicMarket("coinbase").candles(end - 512 * HOUR, end)
        secondary = PublicMarket("kraken").candles(end - HOUR, end)
        cutoff = now()
        MarketDataValidator.validate(primary, cutoff, 512)
        MarketDataValidator.validate(secondary, cutoff, 1)
        a, b = Decimal(primary[-1].close), Decimal(secondary[-1].close)
        if abs(a - b) / ((a + b) / 2) > Decimal(".01"):
          raise ValueError("SOURCE_DIVERGENCE at input origin")
        record_event(
          ledger,
          "AVAILABLE",
          json.dumps(secondary[-1].to_dict()),
          operation="provider_check",
        )
        prediction = generate(ledger, primary, cutoff, TimesFM())
        record_event(
          ledger, "SUCCESS", prediction["prediction_id"], operation="generate"
        )
        print(json.dumps(prediction, indent=2))
    elif args.command == "verify":
      existing = {
        (v["prediction_id"], v["horizon"]) for v in ledger.records("verifications")
      }
      for prediction in ledger.records("predictions"):
        origin = utc(prediction["forecast_origin"])
        mature = min(24, int((now() - origin) / HOUR))
        pending = [
          h
          for h in range(1, mature + 1)
          if (prediction["prediction_id"], h) not in existing
        ]
        if not pending:
          continue
        try:
          start = origin + (min(pending) - 1) * HOUR
          end = origin + max(pending) * HOUR
          sources = {
            name: {c.close_time: c for c in PublicMarket(name).candles(start, end)}
            for name in ("coinbase", "kraken")
          }
          for h in pending:
            target = origin + h * HOUR
            verify(
              ledger,
              prediction["prediction_id"],
              h,
              sources["coinbase"][target],
              sources["kraken"][target],
              now(),
            )
          record_event(
            ledger, "SUCCESS", prediction["prediction_id"], operation="verify"
          )
        except Exception as exc:  # noqa: BLE001 -- Persist boundary failure and return nonzero.
          failures.append(str(exc))
          record_event(
            ledger,
            "VERIFICATION_ERROR",
            f"{prediction['prediction_id']}: {type(exc).__name__}: {exc}",
            operation="verify",
          )
  except Exception as exc:  # noqa: BLE001 -- Persist boundary failure and return nonzero.
    failures.append(str(exc))
    record_event(
      ledger,
      getattr(
        exc,
        "status",
        "MODEL_ERROR" if args.command == "generate" else "OPERATION_ERROR",
      ),
      f"{type(exc).__name__}: {exc}",
      operation=args.command,
    )
  finally:
    ledger.export(args.output_dir / "data/forward-export.json")
    write_report(ledger, args.output_dir / "reports/data.json")
    ledger.close()
  if failures:
    for failure in failures:
      print(failure)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
