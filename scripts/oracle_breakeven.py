"""Oracle break-even analysis for BTC-USD at 1-day, 3-day, and 7-day horizons.

Pure standard library. Evaluates theoretical maximum profitability (perfect
oracle) under realistic trading costs (maker/taker fees, spread, slippage)
and Italian taxation rules (33% capital gains tax on gross gains, non-deductible
fees). If even the oracle cannot beat buy-and-hold, no predictive model can.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from scripts.ohlc_io import Candle, load_csv, validate_series

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default parameters & constants
# ---------------------------------------------------------------------------

TAX_RATE: float = 0.33  # 33% Italian substitute tax on crypto gross capital gains
DEFAULT_HORIZONS: tuple[int, ...] = (1, 3, 7)
DEFAULT_EXTRA_COST: float = 0.0015  # 0.15% = 15 bps round-trip spread + slippage


# ---------------------------------------------------------------------------
# Cost Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostModel:
  """Cost and tax model for a single trade or strategy.

  f: fee per side (entry and exit each pay (1 - f))
  e: round-trip spread and slippage cost
  M: net multiplier = (1 - f)^2 * (1 - e)
  c: round-trip cost = 1 - M
  """

  f: float
  e: float

  @property
  def M(self) -> float:
    return ((1.0 - self.f) ** 2) * (1.0 - self.e)

  @property
  def c(self) -> float:
    return 1.0 - self.M

  @property
  def r_be_pre(self) -> float:
    """Pre-tax break-even return: r such that net cash == 0."""
    return (1.0 / self.M) - 1.0

  @property
  def r_be_tax(self) -> float:
    """After-tax break-even return: r such that net cash - 0.33*max(0, r) == 0."""
    denom = self.M - TAX_RATE
    if denom <= 0:
      return float("inf")
    return (1.0 - self.M) / denom

  def net_cash(self, r: float) -> float:
    """Net cash return before tax: M * (1 + r) - 1."""
    return self.M * (1.0 + r) - 1.0

  def net_after_tax(self, r: float) -> float:
    """Net return after 33% tax on gross gain: M*(1+r) - 1 - 0.33*max(0, r)."""
    return self.net_cash(r) - TAX_RATE * max(0.0, r)

  def p_star_pre(self, m: float) -> float:
    """Pre-tax break-even hit rate: 0.5 + c / (2 * m)."""
    if m <= 0.0:
      return float("inf")
    return 0.5 + (self.c / (2.0 * m))

  def p_star_tax(self, m: float) -> float:
    """After-tax break-even hit rate: 0.5 + c / (1.34 * m)."""
    if m <= 0.0:
      return float("inf")
    return 0.5 + (self.c / (1.34 * m))


@dataclass(frozen=True)
class FeeScenario:
  """Named fee scenario specification."""

  name: str
  f: float
  e: float
  note: str = "to re-verify in official tables on the run date"


def get_default_scenarios(e_extra: float = DEFAULT_EXTRA_COST) -> list[FeeScenario]:
  """Return the 7 standard fee scenarios."""
  return [
    FeeScenario("coinbase_eu_maker", f=0.0025, e=e_extra),
    FeeScenario("coinbase_eu_taker", f=0.0050, e=e_extra),
    FeeScenario("kraken_t1_maker", f=0.0040, e=e_extra),
    FeeScenario("kraken_t1_taker", f=0.0080, e=e_extra),
    FeeScenario("kraken_t2_maker", f=0.0030, e=e_extra),
    FeeScenario("kraken_t2_taker", f=0.0060, e=e_extra),
    # repo_assumption: total round-trip 0.0035, treat as f=0.00175 and e=0
    FeeScenario("repo_assumption", f=0.00175, e=0.0),
  ]


# ---------------------------------------------------------------------------
# Returns computation & Statistics
# ---------------------------------------------------------------------------


def compute_returns(prices: Sequence[float], horizon: int, step: int = 1) -> list[float]:
  """Compute simple returns r = prices[t+H] / prices[t] - 1.

  step = 1: overlapping series (t = 0, 1, 2, ...)
  step = horizon: independent non-overlapping series (t = 0, H, 2H, ...)
  """
  n = len(prices)
  if horizon <= 0:
    raise ValueError(f"Horizon must be positive, got {horizon}")
  if n <= horizon:
    raise ValueError(f"Series length ({n}) must be greater than horizon ({horizon})")

  returns: list[float] = []
  for t in range(0, n - horizon, step):
    p0 = prices[t]
    p1 = prices[t + horizon]
    returns.append((p1 / p0) - 1.0)
  return returns


def quantile(sorted_values: Sequence[float], p: float) -> float:
  """Compute quantile with linear interpolation (p in [0.0, 1.0])."""
  if not sorted_values:
    raise ValueError("Cannot compute quantile of empty sequence")
  if p < 0.0 or p > 1.0:
    raise ValueError(f"Quantile probability must be in [0, 1], got {p}")
  if len(sorted_values) == 1:
    return sorted_values[0]

  idx = p * (len(sorted_values) - 1)
  low = math.floor(idx)
  high = math.ceil(idx)
  weight = idx - low
  return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


@dataclass(frozen=True)
class HorizonReturnStats:
  """Summary statistics of returns for a given horizon."""

  horizon: int
  n_overlapping: int
  n_independent: int
  mean_abs_r: float
  q10: float
  q25: float
  q50: float
  q75: float
  q90: float
  q95: float


def compute_return_stats(
  prices: Sequence[float],
  horizon: int,
) -> tuple[HorizonReturnStats, list[float], list[float]]:
  """Compute return distributions for overlapping and independent sets."""
  over_r = compute_returns(prices, horizon, step=1)
  indep_r = compute_returns(prices, horizon, step=horizon)

  sorted_over = sorted(over_r)
  mean_abs = sum(abs(r) for r in over_r) / len(over_r)

  stats = HorizonReturnStats(
    horizon=horizon,
    n_overlapping=len(over_r),
    n_independent=len(indep_r),
    mean_abs_r=mean_abs,
    q10=quantile(sorted_over, 0.10),
    q25=quantile(sorted_over, 0.25),
    q50=quantile(sorted_over, 0.50),
    q75=quantile(sorted_over, 0.75),
    q90=quantile(sorted_over, 0.90),
    q95=quantile(sorted_over, 0.95),
  )
  return stats, over_r, indep_r


# ---------------------------------------------------------------------------
# Oracle Evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleResult:
  """Evaluation metrics for the oracle strategy under a specific threshold."""

  threshold_name: str  # "pre_tax" or "after_tax"
  threshold_value: float
  trades_count: int
  total_periods: int
  share_tradable: float
  mean_net_after_tax: float
  median_net_after_tax: float
  sum_net_after_tax: float
  sum_per_year: float


def evaluate_oracle(
  returns: Sequence[float],
  cost_model: CostModel,
  threshold: float,
  threshold_name: str,
  years: float,
) -> OracleResult:
  """Evaluate oracle long/flat strategy on a sequence of returns."""
  total = len(returns)
  if total == 0:
    return OracleResult(
      threshold_name=threshold_name,
      threshold_value=threshold,
      trades_count=0,
      total_periods=0,
      share_tradable=0.0,
      mean_net_after_tax=0.0,
      median_net_after_tax=0.0,
      sum_net_after_tax=0.0,
      sum_per_year=0.0,
    )

  tradable_returns = [r for r in returns if r > threshold]
  trades_count = len(tradable_returns)
  share_tradable = trades_count / total

  if trades_count == 0:
    return OracleResult(
      threshold_name=threshold_name,
      threshold_value=threshold,
      trades_count=0,
      total_periods=total,
      share_tradable=0.0,
      mean_net_after_tax=0.0,
      median_net_after_tax=0.0,
      sum_net_after_tax=0.0,
      sum_per_year=0.0,
    )

  net_tax_list = [cost_model.net_after_tax(r) for r in tradable_returns]
  sorted_net_tax = sorted(net_tax_list)

  mean_net = sum(net_tax_list) / trades_count
  median_net = quantile(sorted_net_tax, 0.50)
  sum_net = sum(net_tax_list)
  sum_per_year = sum_net / years if years > 0 else 0.0

  return OracleResult(
    threshold_name=threshold_name,
    threshold_value=threshold,
    trades_count=trades_count,
    total_periods=total,
    share_tradable=share_tradable,
    mean_net_after_tax=mean_net,
    median_net_after_tax=median_net,
    sum_net_after_tax=sum_net,
    sum_per_year=sum_per_year,
  )


@dataclass(frozen=True)
class BuyAndHoldResult:
  """Buy-and-hold benchmark evaluation."""

  r_total: float
  net_cash_total: float
  net_after_tax_total: float
  net_after_tax_per_year: float


def evaluate_buy_and_hold(
  first_close: float,
  last_close: float,
  cost_model: CostModel,
  years: float,
) -> BuyAndHoldResult:
  """Evaluate single buy-and-hold trade over the entire series."""
  r_bh = (last_close / first_close) - 1.0
  net_cash = cost_model.net_cash(r_bh)
  net_tax = cost_model.net_after_tax(r_bh)
  per_year = net_tax / years if years > 0 else 0.0
  return BuyAndHoldResult(
    r_total=r_bh,
    net_cash_total=net_cash,
    net_after_tax_total=net_tax,
    net_after_tax_per_year=per_year,
  )


@dataclass(frozen=True)
class ScenarioEvaluation:
  """Complete evaluation for one horizon and one fee scenario."""

  horizon: int
  scenario: FeeScenario
  cost_model: CostModel
  p_star_pre: float
  p_star_tax: float
  stats: HorizonReturnStats
  oracle_pre: OracleResult
  oracle_tax: OracleResult
  buy_and_hold: BuyAndHoldResult


# ---------------------------------------------------------------------------
# Reporting & Output Generation
# ---------------------------------------------------------------------------


def format_p_star(val: float) -> str:
  """Format p_star; print 'unreachable' if >= 1.0."""
  if val >= 1.0 or math.isinf(val):
    return "unreachable"
  return f"{val:.4f}"


def build_markdown_report(
  evaluations: Sequence[ScenarioEvaluation],
  stats_by_horizon: dict[int, HorizonReturnStats],
  meta: dict[str, object],
  years: float,
) -> str:
  """Build the comprehensive research/reports/oracle_breakeven.md report."""
  lines: list[str] = [
    "# BTC Oracle Break-Even Analysis",
    "",
    "## Provenance & Dataset Information",
    "",
    f"- **Data source**: {meta.get('source', 'Coinbase Exchange REST API')}",
    f"- **Product**: {meta.get('product', 'BTC-USD')}",
    f"- **Granularity**: {meta.get('granularity', 86400)} s (daily OHLC)",
    f"- **Fetched at (UTC)**: {meta.get('fetched_at_utc', 'N/A')}",
    f"- **Rows (daily candles)**: {meta.get('rows', len(evaluations))}",
    f"- **First date**: {meta.get('first_date', 'N/A')}",
    f"- **Last date**: {meta.get('last_date', 'N/A')}",
    f"- **Observation span**: {years:.2f} years",
    f"- **CSV SHA-256**: `{meta.get('sha256', 'N/A')}`",
    "",
    "## Model Assumptions and Limitations",
    "",
    "1. **Execution at Daily Close**: Entry and exit are assumed exactly at daily Close.",
    "   *Limitation*: Known OPTIMISTIC simplification; in live trading, execution occurs",
    "   at the bid/ask spread and with execution delay.",
    "2. **Italian Taxation (33%)**: Capital gains tax is modeled at 33% on gross price gains.",
    "   *Limitation*: Fees and trading costs are treated as NON-DEDUCTIBLE.",
    "   *Notice*: Assumption to be confirmed by an Italian qualified accountant.",
    "3. **Tax Aggregation**: For multi-period sums, single-trade tax equals aggregated tax",
    "   because the oracle trades exclusively periods with positive gains (r > threshold > 0).",
    "   All losses offset gains within the year (optimistic simplification).",
    "4. **Oracle Capital**: Independent trades sum net after-tax returns assuming a constant",
    "   capital of 1 unit per trade (uncompounded arithmetic sum / years).",
    "5. **Comparability**: The oracle sum per year and Buy-and-Hold per year are strictly",
    "   comparable only as 'total return on initial capital 1'.",
    "6. **Break-Even Hit Rate ($p^*$)**: APPROXIMATION assuming symmetric moves of mean size",
    "   $m = \\text{mean}(|r|)$, linearized costs, and loss offsetting.",
    "",
    "## Empirical Return Distributions by Horizon (Overlapping Set)",
    "",
    "| Horizon (days) | Samples (Overlapping) | Samples (Independent) | Mean(|r|) | Q10 | Q25 | Q50 (Median) | Q75 | Q90 | Q95 |",
    "|---|---|---|---|---|---|---|---|---|---|",
  ]

  for h in sorted(stats_by_horizon.keys()):
    st = stats_by_horizon[h]
    lines.append(
      f"| {st.horizon}d | {st.n_overlapping} | {st.n_independent} | "
      f"{st.mean_abs_r:.4f} | {st.q10:+.4f} | {st.q25:+.4f} | {st.q50:+.4f} | "
      f"{st.q75:+.4f} | {st.q90:+.4f} | {st.q95:+.4f} |"
    )

  lines.extend([
    "",
    "## Oracle Results by Horizon",
    "",
  ])

  for h in sorted(stats_by_horizon.keys()):
    lines.extend([
      f"### Horizon {h} Day(s)",
      "",
      "| Scenario | Fee f | Extra e | c (Round-trip) | $r_{be,pre}$ | $r_{be,tax}$ | $p^*_{pre}$ | $p^*_{tax}$ | Tradable % (Tax) | Mean Net (Tax) | Med Net (Tax) | Oracle Indep Sum/Yr (Tax) | B&H Net/Yr |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ])
    h_evals = [e for e in evaluations if e.horizon == h]
    for ev in h_evals:
      sc = ev.scenario
      cm = ev.cost_model
      ot = ev.oracle_tax
      bh = ev.buy_and_hold
      lines.append(
        f"| {sc.name} | {sc.f:.4f} | {sc.e:.4f} | {cm.c:.4f} | "
        f"{cm.r_be_pre:.4f} | {cm.r_be_tax:.4f} | "
        f"{format_p_star(ev.p_star_pre)} | {format_p_star(ev.p_star_tax)} | "
        f"{ot.share_tradable * 100:.1f}% | {ot.mean_net_after_tax:.4f} | "
        f"{ot.median_net_after_tax:.4f} | {ot.sum_per_year:+.4f} | "
        f"{bh.net_after_tax_per_year:+.4f} |"
      )
    lines.append("")

  return "\n".join(lines) + "\n"


def write_csv_report(
  evaluations: Sequence[ScenarioEvaluation],
  path: Path,
) -> None:
  """Write research/reports/oracle_breakeven.csv (1 row per horizon x scenario)."""
  path.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = [
    "horizon",
    "scenario",
    "fee_per_side_f",
    "extra_cost_e",
    "cost_multiplier_M",
    "round_trip_c",
    "r_be_pre",
    "r_be_tax",
    "p_star_pre",
    "p_star_tax",
    "share_tradable_pre",
    "share_tradable_tax",
    "mean_net_after_tax_pre",
    "median_net_after_tax_pre",
    "mean_net_after_tax_tax",
    "median_net_after_tax_tax",
    "oracle_indep_sum_per_year_pre",
    "oracle_indep_sum_per_year_tax",
    "buy_hold_net_after_tax_total",
    "buy_hold_net_after_tax_per_year",
    "note",
  ]
  with path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for ev in evaluations:
      writer.writerow({
        "horizon": ev.horizon,
        "scenario": ev.scenario.name,
        "fee_per_side_f": repr(ev.scenario.f),
        "extra_cost_e": repr(ev.scenario.e),
        "cost_multiplier_M": repr(ev.cost_model.M),
        "round_trip_c": repr(ev.cost_model.c),
        "r_be_pre": repr(ev.cost_model.r_be_pre),
        "r_be_tax": repr(ev.cost_model.r_be_tax),
        "p_star_pre": format_p_star(ev.p_star_pre),
        "p_star_tax": format_p_star(ev.p_star_tax),
        "share_tradable_pre": repr(ev.oracle_pre.share_tradable),
        "share_tradable_tax": repr(ev.oracle_tax.share_tradable),
        "mean_net_after_tax_pre": repr(ev.oracle_pre.mean_net_after_tax),
        "median_net_after_tax_pre": repr(ev.oracle_pre.median_net_after_tax),
        "mean_net_after_tax_tax": repr(ev.oracle_tax.mean_net_after_tax),
        "median_net_after_tax_tax": repr(ev.oracle_tax.median_net_after_tax),
        "oracle_indep_sum_per_year_pre": repr(ev.oracle_pre.sum_per_year),
        "oracle_indep_sum_per_year_tax": repr(ev.oracle_tax.sum_per_year),
        "buy_hold_net_after_tax_total": repr(ev.buy_and_hold.net_after_tax_total),
        "buy_hold_net_after_tax_per_year": repr(ev.buy_and_hold.net_after_tax_per_year),
        "note": ev.scenario.note,
      })


def build_research_readme() -> str:
  """Build research/README.md describing usage, methodology, and limitations."""
  return """# BTC Profitability Oracle & Data Pipeline

This research module investigates the mathematical ceiling of profitability for
trading Bitcoin (BTC-USD) after realistic trading execution costs and Italian
capital gains taxes across 1-day, 3-day, and 7-day prediction horizons.

## Core Principle: The Oracle Ceiling

An **oracle** possesses perfect foresight of future price returns.
If even a perfect oracle with 100% directional accuracy cannot comfortably
outperform buy-and-hold after deducting exchange fees, slippage, and taxes,
then **no machine learning model** (including TimesFM, Chronos, ARIMA, etc.)
can be profitable in live trading.

## How to Run

### Step 1: Download Historical Daily OHLC Candles
```bash
python -m scripts.download_daily_ohlc --years 5 --out research/data/btc_usd_1d.csv
```
Downloads 5 years of daily OHLC candles from Coinbase Exchange public REST API,
excluding the current incomplete day, and saves the data alongside a verified
SHA-256 metadata file (`research/data/btc_usd_1d.meta.json`).

### Step 2: Compute Oracle Break-Even and Payoff Ceilings
```bash
python -m scripts.oracle_breakeven \\
  --csv research/data/btc_usd_1d.csv \\
  --e 0.0015 \\
  --out-md research/reports/oracle_breakeven.md \\
  --out-csv research/reports/oracle_breakeven.csv
```

### Running Tests
```bash
python -m pytest tests/test_download_daily_ohlc.py tests/test_oracle_breakeven.py -v
```

## Cost and Tax Formulas

- **Cost multiplier**: $M = (1 - f)^2 \\times (1 - e)$
  - $f$: maker or taker fee per side (entry and exit)
  - $e$: round-trip spread and slippage (default 0.15% = 15 bps)
- **Round-trip cost**: $c = 1 - M$
- **Break-even return (pre-tax)**: $r_{be,pre} = \\frac{1}{M} - 1$
- **Break-even return (after-tax)**: $r_{be,tax} = \\frac{1 - M}{M - 0.33}$
- **Single-trade net after tax**: $\\text{net} = M \\times (1 + r) - 1 - 0.33 \\times \\max(0, r)$
- **Required hit rate approximation**:
  - Pre-tax: $p^*_{pre} = 0.5 + \\frac{c}{2 \\cdot m}$
  - After-tax: $p^*_{tax} = 0.5 + \\frac{c}{1.34 \\cdot m}$
  *(where $m = \\text{mean}(|r|)$)*

## Important Assumptions & Limitations

1. **Close-to-Close Execution**: Trades enter and exit at the daily Close. In practice,
   market execution occurs at current order book quotes with liquidity constraints.
2. **Italian Taxation**: 33% substitute tax on gross price appreciation. Exchange fees
   are assumed non-deductible (pending consultation with an Italian tax professional).
3. **Loss Offsetting**: Within-year gross gains and losses offset each other.
"""


# ---------------------------------------------------------------------------
# CLI & Execution
# ---------------------------------------------------------------------------


def run_analysis(
  csv_path: Path,
  e_extra: float = DEFAULT_EXTRA_COST,
) -> tuple[list[ScenarioEvaluation], dict[int, HorizonReturnStats], dict[str, object], float]:
  """Run complete oracle break-even evaluation over all horizons and scenarios."""
  candles = load_csv(csv_path)
  validate_series(candles)

  meta_path = csv_path.parent / (csv_path.stem + ".meta.json")
  meta: dict[str, object] = {}
  if meta_path.exists():
    try:
      meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
      meta = {}

  # Observation span in years
  span_days = (candles[-1].date_utc - candles[0].date_utc).days
  years = span_days / 365.0 if span_days > 0 else 1.0

  close_prices = [c.close for c in candles]
  first_close = close_prices[0]
  last_close = close_prices[-1]

  scenarios = get_default_scenarios(e_extra)
  stats_by_horizon: dict[int, HorizonReturnStats] = {}
  evaluations: list[ScenarioEvaluation] = []

  for h in DEFAULT_HORIZONS:
    stats, over_r, indep_r = compute_return_stats(close_prices, h)
    stats_by_horizon[h] = stats

    for sc in scenarios:
      cm = CostModel(f=sc.f, e=sc.e)
      p_pre = cm.p_star_pre(stats.mean_abs_r)
      p_tax = cm.p_star_tax(stats.mean_abs_r)

      # Evaluate oracle on independent set (constant capital 1 per trade)
      res_pre = evaluate_oracle(indep_r, cm, cm.r_be_pre, "pre_tax", years)
      res_tax = evaluate_oracle(indep_r, cm, cm.r_be_tax, "after_tax", years)

      bh = evaluate_buy_and_hold(first_close, last_close, cm, years)

      evaluations.append(
        ScenarioEvaluation(
          horizon=h,
          scenario=sc,
          cost_model=cm,
          p_star_pre=p_pre,
          p_star_tax=p_tax,
          stats=stats,
          oracle_pre=res_pre,
          oracle_tax=res_tax,
          buy_and_hold=bh,
        )
      )

  return evaluations, stats_by_horizon, meta, years


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
  parser = argparse.ArgumentParser(description="BTC Oracle Break-Even Analysis.")
  parser.add_argument(
    "--csv",
    default="research/data/btc_usd_1d.csv",
    help="Path to input daily OHLC CSV (default: research/data/btc_usd_1d.csv)",
  )
  parser.add_argument(
    "--e",
    type=float,
    default=DEFAULT_EXTRA_COST,
    help="Extra spread+slippage round-trip cost (default: 0.0015)",
  )
  parser.add_argument(
    "--out-md",
    default="research/reports/oracle_breakeven.md",
    help="Path for markdown report output (default: research/reports/oracle_breakeven.md)",
  )
  parser.add_argument(
    "--out-csv",
    default="research/reports/oracle_breakeven.csv",
    help="Path for CSV report output (default: research/reports/oracle_breakeven.csv)",
  )
  args = parser.parse_args()

  csv_path = Path(args.csv)
  if not csv_path.exists():
    print(f"ERROR: Input CSV {csv_path} does not exist", file=sys.stderr)
    sys.exit(1)

  evaluations, stats_by_horizon, meta, years = run_analysis(csv_path, e_extra=args.e)

  # Write reports
  md_report = build_markdown_report(evaluations, stats_by_horizon, meta, years)
  out_md_path = Path(args.out_md)
  out_md_path.parent.mkdir(parents=True, exist_ok=True)
  out_md_path.write_text(md_report, encoding="utf-8")

  out_csv_path = Path(args.out_csv)
  write_csv_report(evaluations, out_csv_path)

  readme_path = Path("research/README.md")
  readme_path.parent.mkdir(parents=True, exist_ok=True)
  readme_path.write_text(build_research_readme(), encoding="utf-8")

  # Console summary
  print("=" * 78)
  print("BTC ORACLE BREAK-EVEN SUMMARY")
  print("=" * 78)
  print(f"Input CSV       : {csv_path}")
  print(f"Observation Span: {years:.2f} years")
  print(f"Markdown Report : {out_md_path}")
  print(f"CSV Report      : {out_csv_path}")
  print(f"README          : {readme_path}")
  print("-" * 78)
  print(f"{'Horizon':<8} {'Scenario':<20} {'RoundTrip c':<12} {'r_be_tax':<10} {'p*_tax':<10} {'Tradable%':<10} {'Oracle/Yr':<12} {'B&H/Yr':<10}")
  print("-" * 78)
  for ev in evaluations:
    ot = ev.oracle_tax
    print(
      f"{ev.horizon}d{'':<6} "
      f"{ev.scenario.name:<20} "
      f"{ev.cost_model.c:.4f}{'':<6} "
      f"{ev.cost_model.r_be_tax:.4f}{'':<4} "
      f"{format_p_star(ev.p_star_tax):<10} "
      f"{ot.share_tradable * 100:.1f}%{'':<5} "
      f"{ot.sum_per_year:+.4f}{'':<5} "
      f"{ev.buy_and_hold.net_after_tax_per_year:+.4f}"
    )
  print("=" * 78)


if __name__ == "__main__":
  main()
