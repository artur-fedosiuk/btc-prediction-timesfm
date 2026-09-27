"""CI attestation, isolated updates, and recovery of Git-persisted SQLite state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from .contracts import CHECKPOINT, REVISION, ForecastValidator, digest
from .pipeline import environment, now
from .storage import Ledger, atomic_write

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "reports/remote"
STATE_FILES = ("data/forward.sqlite", "data/forward-export.json", "reports/data.json")


def code_hash(root: Path = ROOT) -> str:
  files = []
  for folder in ("evaluation", "src", "tests"):
    files.extend((root / folder).rglob("*.py"))
  files.extend((root / "requirements").glob("btc-*.txt"))
  files.extend(
    [root / "pyproject.toml", root / ".github/workflows/bitcoin-evaluation.yml"]
  )
  return digest(
    {
      str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
      for p in sorted(files)
    }
  )


def require_gate() -> dict:
  gate = json.loads((EVIDENCE / "quality-gate.json").read_text())
  if (
    gate["status"] != "PASS"
    or gate["code_hash"] != code_hash()
    or not gate["github_run_id"]
  ):
    raise RuntimeError(
      "QUALITY_GATE_REQUIRED: current code has no successful CI attestation"
    )
  return gate


def recover(root: Path = ROOT) -> dict:
  ledger = Ledger(root / STATE_FILES[0])
  try:
    exported = json.loads((root / STATE_FILES[1]).read_text())
    for table, records in exported.items():
      if table == "schema_version":
        continue
      if ledger.records(table) != records:
        raise RuntimeError(f"STORAGE_ERROR: export mismatch in {table}")
    predictions = ledger.records("predictions")
    verified = {
      (v["prediction_id"], v["horizon"]) for v in ledger.records("verifications")
    }
    return {
      "status": "RECOVERED",
      "predictions": [
        {
          "prediction_id": p["prediction_id"],
          "input_hash": p["input_hash"],
          "forecast_hash": p["forecast_hash"],
          "forecast_origin": p["forecast_origin"],
          "generated_at": p["generated_at"],
          "targets": [
            {
              "horizon": h,
              "target_timestamp": p["forecast_points"][h - 1]["target_timestamp"],
              "status": "VERIFIED"
              if (p["prediction_id"], h) in verified
              else "PENDING",
            }
            for h in (1, 4, 8, 12, 24)
          ],
        }
        for p in predictions
      ],
      "verification_count": len(verified),
    }
  finally:
    ledger.close()


def isolated_operation(command: str, root: Path = ROOT, runner=subprocess.run) -> dict:
  """Publish all derived outputs only if the complete operation validates successfully."""
  before = recover(root)
  with tempfile.TemporaryDirectory(prefix="btc-forward-") as temporary:
    stage = Path(temporary)
    (stage / "data").mkdir()
    shutil.copy2(root / STATE_FILES[0], stage / STATE_FILES[0])
    result = runner(
      [
        os.sys.executable,
        "-m",
        "evaluation.forward.cli",
        command,
        "--ledger",
        str(stage / STATE_FILES[0]),
        "--output-dir",
        str(stage),
        "--ci-evidence",
        str(root / "reports/provider-checks-ci.json"),
      ],
      cwd=root,
      check=False,
    )
    if result.returncode:
      raise RuntimeError(
        f"{command.upper()}_ERROR: staged state discarded; exit={result.returncode}"
      )
    after = recover(stage)
    old = {p["prediction_id"]: p for p in before["predictions"]}
    new = {p["prediction_id"]: p for p in after["predictions"]}
    for identity, prediction in old.items():
      if identity not in new or any(
        new[identity][k] != prediction[k]
        for k in ("input_hash", "forecast_hash", "forecast_origin", "generated_at")
      ):
        raise RuntimeError("STORAGE_ERROR: existing prediction mutated")
    # Git commit is the remote atomic publication boundary. Push only after all files validate.
    for name in STATE_FILES:
      destination = root / name
      destination.parent.mkdir(parents=True, exist_ok=True)
      temporary_file = destination.with_suffix(destination.suffix + ".staged")
      shutil.copy2(stage / name, temporary_file)
      os.replace(temporary_file, destination)
    return {"before": before, "after": after}


def smoke() -> dict:
  from .model import TimesFM, preflight

  resources = preflight()
  model = TimesFM()
  values = [100 + i * 0.01 for i in range(512)]
  origin = now().replace(minute=0, second=0, microsecond=0)
  point, lower, upper = model(values)
  points = ForecastValidator.validate(point, lower, upper, origin)
  result = {
    "status": "PASS",
    "evidence": "REAL CHECKPOINT RUNTIME ONLY; SYNTHETIC INPUT",
    "checkpoint": CHECKPOINT,
    "checkpoint_revision": REVISION,
    "environment": environment("cpu"),
    "resource_preflight": resources,
    "input_values": values,
    "forecast_points": points,
    "model_config": model.metadata,
  }
  atomic_write(EVIDENCE / "model-smoke.json", json.dumps(result, indent=2) + "\n")
  return result


def attest() -> dict:
  tests = []
  for file in ("btc-tests.xml", "model-tests.xml"):
    suite = ET.parse(EVIDENCE / file).getroot().find("testsuite")
    if (
      suite is None
      or int(suite.attrib["tests"]) == 0
      or any(int(suite.attrib[k]) for k in ("errors", "failures", "skipped"))
    ):
      raise RuntimeError(f"QUALITY_GATE_FAILED: {file}")
    tests.append({"file": file, **suite.attrib})
  from .cli import require_ci_evidence

  require_ci_evidence(ROOT / "reports/provider-checks-ci.json")
  model = json.loads((EVIDENCE / "model-smoke.json").read_text())
  if model["status"] != "PASS":
    raise RuntimeError("TIMESFM_SMOKE_REQUIRED")
  result = {
    "status": "PASS",
    "code_hash": code_hash(),
    "github_run_id": os.getenv("GITHUB_RUN_ID"),
    "github_sha": os.getenv("GITHUB_SHA"),
    "tests": tests,
    "environment": environment("cpu"),
  }
  if not result["github_run_id"]:
    raise RuntimeError("CI attestation requires GitHub Actions")
  atomic_write(EVIDENCE / "quality-gate.json", json.dumps(result, indent=2) + "\n")
  return result


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "command",
    choices=["smoke", "attest", "require-gate", "recover", "generate", "verify"],
  )
  args = parser.parse_args()
  EVIDENCE.mkdir(parents=True, exist_ok=True)
  if args.command == "require-gate":
    result = require_gate()
  elif args.command == "attest":
    result = attest()
  elif args.command == "smoke":
    result = smoke()
  else:
    require_gate()
    result = (
      recover() if args.command == "recover" else isolated_operation(args.command)
    )
    record = {
      "command": args.command,
      "timestamp": now().isoformat(),
      "environment": environment("cpu" if args.command == "generate" else "NO_MODEL"),
      "git_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
      ).strip(),
      "cache": {
        "model": os.getenv("MODEL_CACHE_HIT"),
        "pip": os.getenv("PIP_CACHE_HIT"),
      },
      "result": result,
    }
    run = (
      os.getenv("GITHUB_RUN_ID", "LOCAL") + "-" + os.getenv("GITHUB_RUN_ATTEMPT", "1")
    )
    atomic_write(EVIDENCE / "runs" / f"{run}.json", json.dumps(record, indent=2) + "\n")
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
