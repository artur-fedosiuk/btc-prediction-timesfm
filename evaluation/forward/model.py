"""Pinned TimesFM 3.0 adapter; model failures never select another model."""

import importlib.metadata
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from .contracts import CHECKPOINT, REVISION, DataError


def preflight() -> dict:
  if sys.platform == "linux":
    fields = {
      line.split(":")[0]: line.split(":")[1].strip()
      for line in Path("/proc/meminfo").read_text().splitlines()
    }
    available = int(fields["MemAvailable"].split()[0]) * 1024
  elif sys.platform == "darwin":
    text = subprocess.check_output(["vm_stat"], text=True)
    page = int(text.split("page size of ")[1].split(" bytes")[0])
    fields = {
      line.split(":")[0]: int(line.split(":")[1].strip().rstrip("."))
      for line in text.splitlines()[1:]
      if ":" in line
    }
    available = page * (fields.get("Pages free", 0) + fields.get("Pages inactive", 0))
  else:
    raise RuntimeError("Resource preflight unsupported on this platform")
  result = {
    "available_memory_bytes": available,
    "free_disk_bytes": shutil.disk_usage(".").free,
    "minimum_memory_bytes": 4 * 1024**3,
  }
  if (
    available < result["minimum_memory_bytes"]
    or result["free_disk_bytes"] < 3 * 1024**3
  ):
    raise RuntimeError(f"INSUFFICIENT_RESOURCES: {result}")
  return result


class TimesFM:
  def __call__(self, values):
    preflight()
    if importlib.metadata.version("timesfm") != "3.0.1":
      raise RuntimeError("Expected timesfm==3.0.1")
    import numpy as np
    import torch

    from timesfm3 import ModelConfig, TimesFM3Evaluator

    torch.manual_seed(0)
    torch.set_num_threads(2)
    config = ModelConfig(
      checkpoint_path=CHECKPOINT, revision=REVISION, device="cpu", per_core_batch_size=1
    )
    self.metadata = {"model_config": asdict(config)}
    model = TimesFM3Evaluator(config)
    output = next(
      model.predict_batch(
        [np.asarray(values, dtype=np.float32)],
        horizon=24,
        return_quantiles=True,
        use_symmetric_averaging=True,
      )
    )
    quantiles = output.quantiles
    if (
      quantiles.shape != (24, 9)
      or not np.isfinite(quantiles).all()
      or (np.diff(quantiles, axis=1) < 0).any()
    ):
      raise DataError("INVALID_FORECAST", "Invalid complete quantile array")
    return output.forecast.tolist(), quantiles[:, 0].tolist(), quantiles[:, 8].tolist()
