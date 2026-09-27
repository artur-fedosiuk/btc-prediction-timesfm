import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.forward.remote import code_hash, isolated_operation, recover
from evaluation.forward.storage import Ledger


def empty_state(root):
  ledger = Ledger(root / "data/forward.sqlite")
  ledger.export(root / "data/forward-export.json")
  ledger.close()
  (root / "reports").mkdir()
  (root / "reports/data.json").write_text("{}")


def test_second_runner_recovers_empty_state(tmp_path):
  empty_state(tmp_path)
  assert recover(tmp_path) == {
    "status": "RECOVERED",
    "predictions": [],
    "verification_count": 0,
  }


def test_failed_operation_never_publishes_partial_state(tmp_path):
  empty_state(tmp_path)
  before = (tmp_path / "data/forward.sqlite").read_bytes()

  def fail(argv, **kwargs):
    stage = Path(argv[argv.index("--ledger") + 1])
    ledger = Ledger(stage)
    with ledger.db:
      ledger.add("events", "partial", {"id": "partial"})
    ledger.close()
    return SimpleNamespace(returncode=1)

  with pytest.raises(RuntimeError, match="staged state discarded"):
    isolated_operation("generate", tmp_path, runner=fail)
  assert (tmp_path / "data/forward.sqlite").read_bytes() == before
  assert recover(tmp_path)["predictions"] == []


def test_export_disagreement_rejected(tmp_path):
  empty_state(tmp_path)
  export = tmp_path / "data/forward-export.json"
  value = json.loads(export.read_text())
  value["events"] = [{"id": "invented"}]
  export.write_text(json.dumps(value))
  with pytest.raises(RuntimeError, match="export mismatch"):
    recover(tmp_path)


def test_code_hash_tracks_code_but_not_reports(tmp_path):
  for folder in (
    "evaluation",
    "src",
    "tests",
    "requirements",
    ".github/workflows",
    "reports",
  ):
    (tmp_path / folder).mkdir(parents=True)
  for file in (
    "pyproject.toml",
    ".github/workflows/bitcoin-evaluation.yml",
    "evaluation/module.py",
  ):
    (tmp_path / file).write_text("initial")
  initial = code_hash(tmp_path)
  (tmp_path / "reports/data.json").write_text("{}")
  assert code_hash(tmp_path) == initial
  (tmp_path / "evaluation/module.py").write_text("changed")
  assert code_hash(tmp_path) != initial
