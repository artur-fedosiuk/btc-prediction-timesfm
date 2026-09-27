"""Transactional append-only SQLite ledger with readable canonical JSON export."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path

from .contracts import (
  HOUR,
  Candle,
  ForecastValidator,
  MarketDataValidator,
  canonical,
  digest,
  utc,
)

TABLES = (
  "experiments",
  "snapshots",
  "predictions",
  "observations",
  "verifications",
  "events",
)


class IntegrityError(ValueError):
  pass


def atomic_write(path: Path, text: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
  try:
    with os.fdopen(fd, "w") as stream:
      stream.write(text)
      stream.flush()
      os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
      os.fsync(directory)
    finally:
      os.close(directory)
  finally:
    if os.path.exists(temporary):
      os.unlink(temporary)


class Ledger:
  def __init__(self, path: str | Path):
    self.path = Path(path)
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self.db = sqlite3.connect(self.path, timeout=30)
    self.db.execute("PRAGMA foreign_keys=ON")
    self.db.execute("PRAGMA synchronous=FULL")
    version = self.db.execute("PRAGMA user_version").fetchone()[0]
    if version not in (0, 1):
      raise IntegrityError(f"Unsupported schema {version}")
    if version == 0:
      with self.db:
        for table in TABLES:
          relation = ""
          if table == "predictions":
            relation = ", experiment_id TEXT NOT NULL REFERENCES experiments(id), snapshot_id TEXT NOT NULL REFERENCES snapshots(id), origin TEXT NOT NULL, UNIQUE(experiment_id, origin)"
          if table == "verifications":
            relation = ", prediction_id TEXT NOT NULL REFERENCES predictions(id), horizon INTEGER NOT NULL CHECK(horizon BETWEEN 1 AND 24), UNIQUE(prediction_id, horizon)"
          self.db.execute(
            f"CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, payload TEXT NOT NULL, hash TEXT NOT NULL{relation})"
          )
          for action in ("UPDATE", "DELETE"):
            self.db.execute(
              f"CREATE TRIGGER IF NOT EXISTS {table}_{action.lower()} BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT, 'Immutable ledger'); END"
            )
        self.db.execute("PRAGMA user_version=1")
    if self.db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
      raise IntegrityError("SQLite integrity check failed")
    for table in TABLES:
      self.records(table)
    for prediction in self.records("predictions"):
      self.validate_prediction(prediction["prediction_id"], prediction)
    if self.db.execute("PRAGMA foreign_key_check").fetchall():
      raise IntegrityError("Broken foreign key")

  def close(self):
    self.db.close()

  def add(self, table: str, identity: str, payload: dict) -> None:
    if table not in TABLES:
      raise ValueError("Unknown table")
    if table == "predictions":
      self.validate_prediction(identity, payload)
    columns, values = (
      ["id", "payload", "hash"],
      [identity, canonical(payload), digest(payload)],
    )
    if table == "predictions":
      columns += ["experiment_id", "snapshot_id", "origin"]
      values += [
        payload["experiment_id"],
        payload["input_snapshot_ref"],
        payload["forecast_origin"],
      ]
    if table == "verifications":
      columns += ["prediction_id", "horizon"]
      values += [payload["prediction_id"], payload["horizon"]]
    try:
      self.db.execute(
        f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in values)})",
        values,
      )
    except sqlite3.IntegrityError as exc:
      raise IntegrityError(str(exc)) from exc

  def validate_prediction(self, identity: str, payload: dict) -> None:
    if (
      payload["prediction_id"] != identity
      or payload["status"] != "VALID"
      or payload["schema_version"] != 1
    ):
      raise IntegrityError("Invalid prediction identity/status/schema")
    if payload["forecast_frequency"] != "1h" or payload["forecast_horizon"] != 24:
      raise IntegrityError("Invalid horizon contract")
    origin, cutoff, generated = map(
      utc, (payload["forecast_origin"], payload["data_cutoff"], payload["generated_at"])
    )
    if not origin <= cutoff <= generated < origin + HOUR:
      raise IntegrityError("Invalid prediction temporal order")
    expected = ForecastValidator.validate(
      payload["forecast_path"], payload["lower_path"], payload["upper_path"], origin
    )
    if (
      expected != payload["forecast_points"]
      or digest(expected) != payload["forecast_hash"]
    ):
      raise IntegrityError("Invalid forecast hash/points")
    snapshots = {digest(s): s for s in self.records("snapshots")}
    snapshot = snapshots.get(payload["input_snapshot_ref"])
    if snapshot is None or digest(snapshot) != payload["input_hash"]:
      raise IntegrityError("Missing/corrupt snapshot")
    candles = [Candle.from_dict(c) for c in snapshot["candles"]]
    required = payload["manifest"]["forecast_config"]["context_length"]
    if MarketDataValidator.validate(candles, cutoff, required) != origin:
      raise IntegrityError("Snapshot origin mismatch")
    import struct

    values = [
      struct.unpack("<f", struct.pack("<f", float(c.close)))[0] for c in candles
    ]
    if snapshot["input_values"] != values or snapshot["timestamps"] != [
      utc(c.close_time).isoformat() for c in candles
    ]:
      raise IntegrityError("Model input differs from snapshot")
    if (
      payload["provider"] != candles[0].provider
      or payload["venue"] != candles[0].venue
      or payload["symbol"] != "BTC/USD"
      or payload["quote_currency"] != "USD"
    ):
      raise IntegrityError("Prediction identity differs from snapshot")

  def records(self, table: str) -> list[dict]:
    if table not in TABLES:
      raise ValueError("Unknown table")
    result = []
    for identity, raw, checksum in self.db.execute(
      f"SELECT id,payload,hash FROM {table} ORDER BY id"
    ):
      value = json.loads(raw)
      if digest(value) != checksum:
        raise IntegrityError(f"Corrupt {table}/{identity}")
      result.append(value)
    return result

  def export(self, path: Path) -> None:
    atomic_write(
      path,
      canonical({"schema_version": 1, **{t: self.records(t) for t in TABLES}}) + "\n",
    )
