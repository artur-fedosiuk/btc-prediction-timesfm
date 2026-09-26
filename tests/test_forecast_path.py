"""Tests for Phase 1.5 Forecast Path metrics and serialization."""

import json
from evaluation.path_metrics import ForecastPathMetrics
import pytest
import numpy as np

def test_forecast_path_metrics_calculation():
    # 24 steps
    entry_price = 100.0
    forecast_path = [
        101.0, 102.0, 103.0, 104.0, # 1-4
        105.0, 106.0, 107.0, 108.0, # 5-8
        109.0, 110.0, 111.0, 112.0, # 9-12
        113.0, 114.0, 115.0, 116.0, # 13-16
        117.0, 118.0, 119.0, 120.0, # 17-20
        121.0, 122.0, 123.0, 124.0  # 21-24
    ]
    
    assert len(forecast_path) == 24
    
    metrics = ForecastPathMetrics.calculate(entry_price, forecast_path)
    
    # Indexes: t1=0, t4=3, t8=7, t12=11, t24=23
    assert metrics.t1 == 101.0
    assert metrics.t4 == 104.0
    assert metrics.t8 == 108.0
    assert metrics.t12 == 112.0
    assert metrics.t24 == 124.0
    
    assert metrics.return_t1_pct == 1.0
    assert metrics.return_t4_pct == 4.0
    assert metrics.return_t24_pct == 24.0
    
    assert metrics.path_min == 101.0
    assert metrics.path_max == 124.0
    
    assert metrics.min_return_pct == 1.0
    assert metrics.max_return_pct == 24.0
    
    assert metrics.range_pct == 23.0 # max - min (24 - 1)
    
def test_forecast_path_metrics_invalid_length():
    with pytest.raises(ValueError, match="exactly 24 points"):
        ForecastPathMetrics.calculate(100.0, [100.0, 101.0])

def test_json_serialization():
    path = [100.0, 101.234, 102.5]
    serialized = json.dumps([round(x, 2) for x in path])
    assert serialized == "[100.0, 101.23, 102.5]"
    
