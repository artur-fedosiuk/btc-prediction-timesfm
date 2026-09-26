"""
Path metrics calculation for the 24h forecast path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class ForecastPathMetrics:
    """Derived metrics from the 24h forecast path."""
    
    t1: float
    t4: float
    t8: float
    t12: float
    t24: float
    
    return_t1_pct: float
    return_t4_pct: float
    return_t8_pct: float
    return_t12_pct: float
    return_t24_pct: float
    
    path_min: float
    path_max: float
    
    min_return_pct: float
    max_return_pct: float
    
    range_pct: float
    slope: float
    volatility: float
    
    @classmethod
    def calculate(cls, entry_price: float, forecast_path: list[float]) -> "ForecastPathMetrics":
        """Calculate metrics given an entry price and a 24-step forecast path."""
        if len(forecast_path) != 24:
            raise ValueError(f"forecast_path must have exactly 24 points, got {len(forecast_path)}")
            
        t1 = forecast_path[0]
        t4 = forecast_path[3]
        t8 = forecast_path[7]
        t12 = forecast_path[11]
        t24 = forecast_path[23]
        
        path_min = min(forecast_path)
        path_max = max(forecast_path)
        
        def pct(target: float) -> float:
            return (target - entry_price) / entry_price * 100.0
            
        returns = [pct(p) for p in forecast_path]
        
        # Slope (linear regression over the 24 points)
        x = np.arange(24)
        y = np.array(returns)
        slope = float(np.polyfit(x, y, 1)[0])
        
        # Volatility (standard deviation of the returns)
        volatility = float(np.std(returns))
        
        return cls(
            t1=t1,
            t4=t4,
            t8=t8,
            t12=t12,
            t24=t24,
            return_t1_pct=pct(t1),
            return_t4_pct=pct(t4),
            return_t8_pct=pct(t8),
            return_t12_pct=pct(t12),
            return_t24_pct=pct(t24),
            path_min=path_min,
            path_max=path_max,
            min_return_pct=pct(path_min),
            max_return_pct=pct(path_max),
            range_pct=pct(path_max) - pct(path_min),
            slope=slope,
            volatility=volatility,
        )
