import sqlite3

import pytest

from evaluation.forward.storage import IntegrityError, Ledger


def test_immutable_duplicate_and_restart(tmp_path):
  path = tmp_path / "ledger.sqlite"
  ledger = Ledger(path)
  with ledger.db:
    ledger.add("events", "a", {"id": "a", "status": "TEST"})
  with pytest.raises(IntegrityError), ledger.db:
    ledger.add("events", "a", {"id": "a", "status": "OTHER"})
  with pytest.raises(sqlite3.IntegrityError):
    ledger.db.execute("UPDATE events SET payload='{}'")
  with pytest.raises(sqlite3.IntegrityError):
    ledger.db.execute("DELETE FROM events")
  ledger.close()
  ledger = Ledger(path)
  assert ledger.records("events") == [{"id": "a", "status": "TEST"}]
  ledger.close()


def test_transaction_rollback(tmp_path):
  ledger = Ledger(tmp_path / "ledger.sqlite")
  with pytest.raises(RuntimeError), ledger.db:
    ledger.add("events", "a", {"id": "a"})
    raise RuntimeError("simulated crash before commit")
  assert ledger.records("events") == []


def test_unknown_schema(tmp_path):
  path = tmp_path / "ledger.sqlite"
  db = sqlite3.connect(path)
  db.execute("PRAGMA user_version=99")
  db.close()
  with pytest.raises(IntegrityError):
    Ledger(path)


def test_foreign_key(tmp_path):
  ledger = Ledger(tmp_path / "ledger.sqlite")
  with pytest.raises(IntegrityError), ledger.db:
    ledger.add("verifications", "v", {"prediction_id": "missing", "horizon": 24})


def test_process_crash_rolls_back(tmp_path):
  import subprocess
  import sys

  path = tmp_path / "crash.sqlite"
  source = """
import os,sys
from evaluation.forward.storage import Ledger
ledger=Ledger(sys.argv[1])
with ledger.db:
    ledger.add('events','uncommitted',{'id':'uncommitted'})
    os._exit(19)
"""
  result = subprocess.run([sys.executable, "-c", source, str(path)], check=False)
  assert result.returncode == 19
  ledger = Ledger(path)
  assert ledger.records("events") == []


def test_corruption_detected(tmp_path):
  path = tmp_path / "corrupt.sqlite"
  ledger = Ledger(path)
  with ledger.db:
    ledger.add("events", "a", {"id": "a"})
  ledger.close()
  db = sqlite3.connect(path)
  db.execute("DROP TRIGGER events_update")
  db.execute("UPDATE events SET payload='{}'")
  db.commit()
  db.close()
  with pytest.raises(IntegrityError, match="Corrupt"):
    Ledger(path)
