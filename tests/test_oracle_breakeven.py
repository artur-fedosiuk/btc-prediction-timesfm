"""Tests for scripts/oracle_breakeven.py.

Verifies:
1. All reference values specified by the reviewer.
2. A hand-computed 10-day price series with full manual calculations in comments.
3. Cross-check against a naive pure-Python reference implementation.
4. Edge cases: fees larger than gain, horizon > len(data), series with missing gap.
5. Sanity: zero costs and zero tax produces exact sum of positive returns.
6. Integration test of run_analysis on synthetic CSV.
"""
from __future__ import annotations

import csv
import json
import math
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import pytest

from scripts.ohlc_io import Candle, validate_series
from scripts.oracle_breakeven import (
  CostModel,
  HorizonReturnStats,
  compute_return_stats,
  compute_returns,
  evaluate_buy_and_hold,
  evaluate_oracle,
  quantile,
  run_analysis,
)


# ---------------------------------------------------------------------------
# Test 1: All reviewer reference values
# ---------------------------------------------------------------------------


def test_reviewer_reference_values() -> None:
  """Exact verification of all reviewer reference values:
  - f=0.005, e=0.0015 -> M = 0.9885399625, c = 0.0114600375
  - r = 0.02 -> net cash = +0.0083107618 (tolerance 1e-9)
  - r_be_pre = 0.0115928925 ; r_be_tax = 0.0174021900 (tolerance 1e-7)
  - Fees exceed gross gain: r = 0.004 -> net cash = -0.0075058777 ;
    after tax = -0.0088258777 (tax is due on the gross gain even if cash is lost)
  - m = 0.02 with the same c -> p_star_pre = 0.7865009 ; p_star_tax = 0.9276133
    (tolerance 1e-6)
  """
  model = CostModel(f=0.005, e=0.0015, tax_rate=0.33)

  # 1. Cost multiplier and round-trip cost
  assert math.isclose(model.M, 0.9885399625, rel_tol=0.0, abs_tol=1e-12)
  assert math.isclose(model.c, 0.0114600375, rel_tol=0.0, abs_tol=1e-12)

  # 2. Return r = +0.02 -> net cash = +0.0083107618
  net_cash_02 = model.net_cash(0.02)
  assert math.isclose(net_cash_02, 0.0083107618, rel_tol=0.0, abs_tol=1e-9)

  # 3. Break-even thresholds
  assert math.isclose(model.r_be_pre, 0.0115928925, rel_tol=0.0, abs_tol=1e-7)
  assert math.isclose(model.r_be_tax, 0.0174021900, rel_tol=0.0, abs_tol=1e-7)

  # 4. Fees exceed gross gain: r = 0.004
  # Gross gain is positive (+0.004), so 33% tax applies to gross gain (0.004 * 0.33 = 0.00132)
  # even though the net cash trade is negative (-0.0075058777).
  net_cash_004 = model.net_cash(0.004)
  net_tax_004 = model.net_after_tax(0.004)
  assert math.isclose(net_cash_004, -0.0075058777, rel_tol=0.0, abs_tol=1e-9)
  assert math.isclose(net_tax_004, -0.0088258777, rel_tol=0.0, abs_tol=1e-9)

  # 5. Required hit rates for m = 0.02
  m = 0.02
  p_pre = model.p_star_pre(m)
  p_tax = model.p_star_tax(m)
  assert math.isclose(p_pre, 0.7865009, rel_tol=0.0, abs_tol=1e-6)
  assert math.isclose(p_tax, 0.9276133, rel_tol=0.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Test 2: Tiny hand-made price series (10 days) with manual derivations
# ---------------------------------------------------------------------------


def test_hand_computed_ten_day_series() -> None:
  """A 10-day hand-computed price series.

  Prices:
    Day 0: 100.0
    Day 1: 105.0
    Day 2: 102.0
    Day 3: 108.0
    Day 4: 106.0
    Day 5: 112.0
    Day 6: 110.0
    Day 7: 115.0
    Day 8: 112.0
    Day 9: 120.0

  Horizon H = 3.
  Overlapping periods (N - H = 10 - 3 = 7):
    t=0: P[3]/P[0] - 1 = 108/100 - 1 = +0.0800000000
    t=1: P[4]/P[1] - 1 = 106/105 - 1 = +0.0095238095
    t=2: P[5]/P[2] - 1 = 112/102 - 1 = +0.0980392157
    t=3: P[6]/P[3] - 1 = 110/108 - 1 = +0.0185185185
    t=4: P[7]/P[4] - 1 = 115/106 - 1 = +0.0849056604
    t=5: P[8]/P[5] - 1 = 112/112 - 1 =  0.0000000000
    t=6: P[9]/P[6] - 1 = 120/110 - 1 = +0.0909090909

  Independent periods (indices t = 0, 3, 6; total 3 periods):
    t=0: 108/100 - 1 = 0.0800000000
    t=3: 110/108 - 1 = 2/108 = 0.0185185185...
    t=6: 120/110 - 1 = 10/110 = 0.0909090909...

  Cost parameters: f = 0.005, e = 0.0015
    M = 0.9885399625
    r_be_pre = 0.0115928925...
    r_be_tax = 0.0174021900...

  Independent trades evaluation against after-tax threshold (r > 0.01740219):
    Trade 1 (r = 0.08):
      net_cash = 0.9885399625 * 1.08 - 1 = +0.0676231595
      tax = 0.33 * 0.08 = 0.0264000000
      net_after_tax = 0.0676231595 - 0.0264 = +0.0412231595
    Trade 2 (r = 2/108 = 0.0185185185... > 0.0174021900):
      net_cash = 0.9885399625 * (55/54) - 1 = +0.0068462585
      tax = 0.33 * (1/54) = 0.0061111111
      net_after_tax = 0.0068462585 - 0.0061111111 = +0.0007351474
    Trade 3 (r = 1/11 = 0.0909090909... > 0.0174021900):
      net_cash = 0.9885399625 * (12/11) - 1 = +0.0784072318
      tax = 0.33 * (1/11) = 0.0300000000
      net_after_tax = 0.0784072318 - 0.0300000000 = +0.0484072318

    Tradable trades: 3 / 3 -> share_tradable = 1.0 (100%)
    Sum net after tax = 0.0412231595 + 0.0007351474 + 0.0484072318 = 0.0903655387
    Mean net after tax = 0.0903655387 / 3 = 0.0301218462
  """
  prices = [100.0, 105.0, 102.0, 108.0, 106.0, 112.0, 110.0, 115.0, 112.0, 120.0]
  model = CostModel(f=0.005, e=0.0015, tax_rate=0.33)

  # Check overlapping returns
  over_r = compute_returns(prices, horizon=3, step=1)
  assert len(over_r) == 7
  assert math.isclose(over_r[0], 0.08, abs_tol=1e-9)
  assert math.isclose(over_r[1], 106.0 / 105.0 - 1.0, abs_tol=1e-9)
  assert math.isclose(over_r[2], 112.0 / 102.0 - 1.0, abs_tol=1e-9)
  assert math.isclose(over_r[3], 2.0 / 108.0, abs_tol=1e-9)
  assert math.isclose(over_r[4], 9.0 / 106.0, abs_tol=1e-9)
  assert math.isclose(over_r[5], 0.0, abs_tol=1e-9)
  assert math.isclose(over_r[6], 10.0 / 110.0, abs_tol=1e-9)

  # Check independent returns
  indep_r = compute_returns(prices, horizon=3, step=3)
  assert len(indep_r) == 3
  assert math.isclose(indep_r[0], 0.08, abs_tol=1e-9)
  assert math.isclose(indep_r[1], 2.0 / 108.0, abs_tol=1e-9)
  assert math.isclose(indep_r[2], 10.0 / 110.0, abs_tol=1e-9)

  # Evaluate oracle on independent set with after-tax threshold
  oracle_res = evaluate_oracle(indep_r, model, model.r_be_tax, "after_tax", years=1.0)
  assert oracle_res.trades_count == 3
  assert oracle_res.total_periods == 3
  assert math.isclose(oracle_res.share_tradable, 1.0, abs_tol=1e-9)

  # Verify exact hand-calculated sum and mean
  expected_sum = 0.0412231595 + 0.000735147361111 + 0.0484072318181817
  assert math.isclose(oracle_res.sum_net_after_tax, expected_sum, abs_tol=1e-8)
  assert math.isclose(oracle_res.mean_net_after_tax, expected_sum / 3.0, abs_tol=1e-8)

  # Check Buy-and-Hold over the entire series (from 100.0 to 120.0)
  # r_bh = 120/100 - 1 = 0.20
  # net_cash = 0.9885399625 * 1.20 - 1 = 0.1862479550
  # tax = 0.33 * 0.20 = 0.0660000000
  # net_after_tax = 0.1862479550 - 0.066 = 0.1202479550
  bh_res = evaluate_buy_and_hold(100.0, 120.0, model, years=1.0)
  assert math.isclose(bh_res.r_total, 0.20, abs_tol=1e-9)
  expected_bh_net = model.M * 1.20 - 1.0 - 0.33 * 0.20
  assert math.isclose(bh_res.net_after_tax_total, expected_bh_net, abs_tol=1e-9)
  assert math.isclose(bh_res.net_after_tax_total, 0.1202479550, abs_tol=1e-8)


# ---------------------------------------------------------------------------
# Test 3: Cross-check against naive pure-Python loop
# ---------------------------------------------------------------------------


def test_cross_check_against_naive_loop() -> None:
  """Production evaluate_oracle and compute_returns must match a naive loop
  implementation on a seeded synthetic random walk.
  """
  rng = random.Random(42)
  p = 50_000.0
  prices = [p]
  for _ in range(200):
    p *= 1.0 + rng.uniform(-0.04, 0.04)
    prices.append(p)

  model = CostModel(f=0.0040, e=0.0015, tax_rate=0.33)
  years = 200 / 365.0

  for h in (1, 3, 7):
    # 1. Naive loop for returns
    naive_indep_r: list[float] = []
    idx = 0
    while idx + h < len(prices):
      naive_indep_r.append((prices[idx + h] / prices[idx]) - 1.0)
      idx += h

    prod_indep_r = compute_returns(prices, horizon=h, step=h)
    assert len(naive_indep_r) == len(prod_indep_r)
    for r_naive, r_prod in zip(naive_indep_r, prod_indep_r, strict=True):
      assert math.isclose(r_naive, r_prod, abs_tol=1e-12)

    # 2. Naive loop for oracle after-tax sum
    threshold = model.r_be_tax
    naive_tradable = [r for r in naive_indep_r if r > threshold]
    naive_net_after_tax: list[float] = []
    for r in naive_tradable:
      net_cash = model.M * (1.0 + r) - 1.0
      tax = 0.33 * max(0.0, r)
      naive_net_after_tax.append(net_cash - tax)

    naive_sum = sum(naive_net_after_tax)
    naive_mean = naive_sum / len(naive_tradable) if naive_tradable else 0.0

    prod_result = evaluate_oracle(prod_indep_r, model, threshold, "after_tax", years)
    assert prod_result.trades_count == len(naive_tradable)
    assert math.isclose(prod_result.sum_net_after_tax, naive_sum, abs_tol=1e-12)
    assert math.isclose(prod_result.mean_net_after_tax, naive_mean, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# Test 4: Edge cases
# ---------------------------------------------------------------------------


def test_edge_case_fees_exceed_any_gain_no_trades() -> None:
  """When fees are excessively high (e.g. 50%), no period has r > threshold.

  Oracle trades 0 periods and returns zeros without crashing.
  """
  model = CostModel(f=0.50, e=0.10)  # Huge fees
  assert model.r_be_tax > 1.0

  # Small price changes [-0.01, +0.02]
  prices = [100.0, 101.0, 102.0, 101.5, 102.0]
  indep_r = compute_returns(prices, horizon=1, step=1)

  res = evaluate_oracle(indep_r, model, model.r_be_tax, "after_tax", years=1.0)
  assert res.trades_count == 0
  assert res.share_tradable == 0.0
  assert res.mean_net_after_tax == 0.0
  assert res.median_net_after_tax == 0.0
  assert res.sum_net_after_tax == 0.0
  assert res.sum_per_year == 0.0


def test_edge_case_horizon_longer_than_series_raises() -> None:
  """Requesting a horizon >= series length must raise ValueError."""
  prices = [100.0, 105.0]  # length 2
  with pytest.raises(ValueError, match="Series length .* must be greater than horizon"):
    compute_returns(prices, horizon=2)

  with pytest.raises(ValueError, match="Horizon must be positive"):
    compute_returns(prices, horizon=0)


def test_edge_case_series_with_gap_rejected_by_validate_series() -> None:
  """A series missing a calendar day must be rejected by validate_series."""
  d0 = date(2023, 1, 1)
  d2 = date(2023, 1, 3)  # Missing 2023-01-02
  c0 = Candle(d0, 100.0, 101.0, 99.0, 100.5, 10.0)
  c2 = Candle(d2, 100.5, 102.0, 100.0, 101.5, 12.0)

  with pytest.raises(ValueError, match="Missing calendar days: 2023-01-02"):
    validate_series([c0, c2])


# ---------------------------------------------------------------------------
# Test 5: Real sanity check (f=0, e=0, tax_rate=0)
# ---------------------------------------------------------------------------


def test_sanity_zero_costs_zero_tax_equals_sum_of_positive_r() -> None:
  """With f=0, e=0, tax_rate=0, the oracle sum must exactly equal the sum of
  all positive returns.
  """
  zero_model = CostModel(f=0.0, e=0.0, tax_rate=0.0)
  assert zero_model.M == 1.0
  assert zero_model.c == 0.0
  assert zero_model.r_be_pre == 0.0
  assert zero_model.r_be_tax == 0.0

  # Arbitrary returns with positive and negative numbers
  returns = [-0.05, 0.03, -0.01, 0.08, 0.0, -0.02, 0.04]
  threshold = 0.0

  # With f=0, e=0, tax_rate=0:
  # net_cash = 1.0 * (1 + r) - 1 = r
  # net_after_tax = r - 0 = r
  expected_sum = sum(r for r in returns if r > 0.0)

  # evaluate_oracle with custom threshold 0.0 and zero model
  oracle_res = evaluate_oracle(returns, zero_model, threshold, "after_tax", years=1.0)
  assert math.isclose(oracle_res.sum_net_after_tax, expected_sum, abs_tol=1e-15)
  assert oracle_res.trades_count == 3  # three positive returns (0.03, 0.08, 0.04)
  assert oracle_res.total_periods == 7


# ---------------------------------------------------------------------------
# Test 6: Integration test of run_analysis on synthetic CSV
# ---------------------------------------------------------------------------


def test_integration_run_analysis_synthetic(tmp_path: Path) -> None:
  """Integration test: run_analysis on a small synthetic CSV (60 days, valid OHLC)."""
  import json

  # Generate 60 days of valid OHLC data with known pattern
  from datetime import date, timedelta

  from scripts.ohlc_io import Candle, save_csv
  from scripts.oracle_breakeven import run_analysis

  # Create prices that go up and down
  base = 50000.0
  prices = []
  for i in range(60):
    if i < 20:
      base *= 1.01  # uptrend
    elif i < 40:
      base *= 0.99  # downtrend
    else:
      base *= 1.005  # slight uptrend
    prices.append(base)

  # Build candles
  start_date = date(2023, 1, 1)
  candles = []
  for i, close in enumerate(prices):
    d = start_date + timedelta(days=i)
    # Create valid OHLC: high >= close, low <= close, etc.
    high = close * 1.02
    low = close * 0.98
    open_ = close * 0.99
    candles.append(Candle(d, open_, high, low, close, 100.0))

  # Write CSV using save_csv (correct format)
  csv_path = tmp_path / "btc_usd_1d.csv"
  save_csv(candles, csv_path)

  # Write meta
  meta_path = tmp_path / "btc_usd_1d.meta.json"
  meta_path.write_text(json.dumps({
    "source": "synthetic_test",
    "product": "BTC-USD",
    "granularity": 86400,
    "fetched_at_utc": "2024-01-01T00:00:00Z",
    "rows": 60,
    "first_date": "2023-01-01",
    "last_date": "2023-03-01",
    "sha256": "test"
  }))

  # Run analysis
  evaluations, stats_by_horizon, meta, years = run_analysis(csv_path)

  # Verify structure
  assert len(evaluations) > 0
  # Should have 5 horizons x 7 scenarios = 35 evaluations
  assert len(evaluations) == 35
  assert set(stats_by_horizon.keys()) == {1, 3, 7, 14, 30}
  assert years > 0

  # Check each evaluation has the right structure
  for ev in evaluations:
    assert ev.horizon in {1, 3, 7, 14, 30}
    assert ev.scenario.name in [
      "coinbase_eu_maker", "coinbase_eu_taker",
      "kraken_t1_maker", "kraken_t1_taker",
      "kraken_t2_maker", "kraken_t2_taker",
      "repo_assumption"
    ]
    assert ev.cost_model.f > 0
    assert ev.cost_model.e >= 0
    assert ev.cost_model.tax_rate == 0.33
    assert 0 <= ev.oracle_tax.share_tradable <= 1
    assert ev.stats.n_independent > 0
