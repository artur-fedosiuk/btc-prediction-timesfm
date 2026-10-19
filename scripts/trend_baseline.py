"""Trend-following baseline strategy for BTC-USD.

At the close of day t, if close[t] > SMA(N)[t] (mean of the last N closes including t),
be long from close[t]; otherwise flat.

A trade is a run of long days: entry at the close of its first day, exit at the close
of the first non-long day (or the last day).

Cost: each trade pays M = (1-f)^2*(1-e), same scenarios as the oracle.
Tax: 33% on the sum of gross trade returns, losses offsetting gains, fees not deductible.
       tax = 0.33*max(0, sum(r_i)). Uncompounded and approximate: say so in the report.

For N in {50, 100, 200}, scenarios coinbase_eu_maker and coinbase_eu_taker, report:
trades, hit rate, total net after tax, max drawdown of the equity curve
(compound M*(1+r_i) per trade), and buy-and-hold on the same period.
"""

from __future__ import annotations

import argparse
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

DEFAULT_TAX_RATE: float = 0.33
DEFAULT_SMAS: tuple[int, ...] = (50, 100, 200)
DEFAULT_EXTRA_COST: float = 0.0015


# ---------------------------------------------------------------------------
# Cost Model (same as oracle)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostModel:
  """Cost and tax model for a single trade or strategy.

  f: fee per side (entry and exit each pay (1 - f))
  e: round-trip spread and slippage cost
  tax_rate: capital gains tax rate on gross gains (default 0.33)
  M: net multiplier = (1 - f)^2 * (1 - e)
  c: round-trip cost = 1 - M
  """

  f: float
  e: float
  tax_rate: float = DEFAULT_TAX_RATE

  @property
  def M(self) -> float:
    return ((1.0 - self.f) ** 2) * (1.0 - self.e)

  @property
  def c(self) -> float:
    return 1.0 - self.M

  def net_cash(self, r: float) -> float:
    """Net cash return before tax: M * (1 + r) - 1."""
    return self.M * (1.0 + r) - 1.0

  def net_after_tax(self, r: float) -> float:
    """Net return after tax on gross gain: M*(1+r) - 1 - tax_rate*max(0, r)."""
    return self.net_cash(r) - self.tax_rate * max(0.0, r)


@dataclass(frozen=True)
class FeeScenario:
  """Named fee scenario specification."""

  name: str
  f: float
  e: float
  note: str = "to re-verify in official tables on the run date"


def get_baseline_scenarios(e_extra: float = DEFAULT_EXTRA_COST) -> list[FeeScenario]:
  """Return the 2 fee scenarios for baseline."""
  return [
    FeeScenario("coinbase_eu_maker", f=0.0025, e=e_extra),
    FeeScenario("coinbase_eu_taker", f=0.0050, e=e_extra),
  ]


# ---------------------------------------------------------------------------
# SMA and Signal Computation
# ---------------------------------------------------------------------------


def compute_sma(prices: Sequence[float], window: int) -> list[float | None]:
  """Compute SMA(window) for each day. Returns None for first window-1 days."""
  n = len(prices)
  sma: list[float | None] = [None] * n
  if n < window:
    return sma

  # Initialize sum of first window prices
  window_sum = sum(prices[:window])
  sma[window - 1] = window_sum / window

  # Rolling update
  for i in range(window, n):
    window_sum += prices[i] - prices[i - window]
    sma[i] = window_sum / window

  return sma


def compute_signals(prices: Sequence[float], window: int) -> list[bool]:
  """Compute long/flat signal: True = long, False = flat.

  At close of day t, if close[t] > SMA(N)[t], be long from close[t].
  """
  sma = compute_sma(prices, window)
  signals = [False] * len(prices)
  for i in range(window - 1, len(prices)):
    if sma[i] is not None:
      signals[i] = prices[i] > sma[i]
  return signals


# ---------------------------------------------------------------------------
# Trade Extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trade:
  """A single trend trade (run of long days)."""

  entry_idx: int
  exit_idx: int
  entry_price: float
  exit_price: float
  gross_return: float  # exit/entry - 1


def extract_trades(
  prices: Sequence[float],
  signals: Sequence[bool],
) -> list[Trade]:
  """Extract trades from price series and signals.

  A trade is a run of long days:
  - Entry at close of first long day
  - Exit at close of first non-long day after the run (or last day)
  """
  trades: list[Trade] = []
  n = len(prices)
  i = 0

  while i < n:
    # Find start of a long run
    if not signals[i]:
      i += 1
      continue

    entry_idx = i
    entry_price = prices[i]

    # Find end of the long run
    j = i + 1
    while j < n and signals[j]:
      j += 1

    exit_idx = j if j < n else n - 1
    exit_price = prices[exit_idx]
    gross_return = (exit_price / entry_price) - 1.0

    trades.append(Trade(
      entry_idx=entry_idx,
      exit_idx=exit_idx,
      entry_price=entry_price,
      exit_price=exit_price,
      gross_return=gross_return,
    ))

    i = j

  return trades


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineResult:
  """Evaluation metrics for the trend baseline strategy."""

  sma_window: int
  scenario: FeeScenario
  cost_model: CostModel
  trades_count: int
  hit_rate: float
  total_net_after_tax: float
  max_drawdown: float
  equity_curve: list[float]
  buy_hold_return: float
  buy_hold_net_after_tax: float


def evaluate_baseline(
  prices: Sequence[float],
  signals: Sequence[bool],
  cost_model: CostModel,
  scenario: FeeScenario,
  sma_window: int,
) -> BaselineResult:
  """Evaluate trend baseline strategy on price series with given signals."""
  trades = extract_trades(prices, signals)

  if not trades:
    # No trades - flat the whole time
    buy_hold_return = (prices[-1] / prices[0]) - 1.0
    bh_net = cost_model.net_after_tax(buy_hold_return)
    return BaselineResult(
      sma_window=sma_window,
      scenario=scenario,
      cost_model=cost_model,
      trades_count=0,
      hit_rate=0.0,
      total_net_after_tax=0.0,
      max_drawdown=0.0,
      equity_curve=[1.0],
      buy_hold_return=buy_hold_return,
      buy_hold_net_after_tax=bh_net,
    )

  # Compute net returns per trade
  gross_returns = [t.gross_return for t in trades]
  net_cash_returns = [cost_model.net_cash(r) for r in gross_returns]

  # Tax: 33% on sum of gross trade returns (losses offset gains)
  sum_gross = sum(gross_returns)
  tax = cost_model.tax_rate * max(0.0, sum_gross)

  # Total net after tax (uncompounded, approximate)
  sum_net_cash = sum(net_cash_returns)
  total_net_after_tax = sum_net_cash - tax

  # Hit rate: fraction of trades with positive net cash return
  winning_trades = sum(1 for r in net_cash_returns if r > 0)
  hit_rate = winning_trades / len(trades) if trades else 0.0

  # Equity curve (compounded): start at 1.0, multiply by M*(1+r) for each trade
  equity = [1.0]
  for r in gross_returns:
    equity.append(equity[-1] * cost_model.M * (1.0 + r))

  # Max drawdown of equity curve
  peak = equity[0]
  max_dd = 0.0
  for val in equity:
    if val > peak:
      peak = val
    dd = (peak - val) / peak if peak > 0 else 0.0
    if dd > max_dd:
      max_dd = dd

  # Buy-and-hold on same period
  buy_hold_return = (prices[-1] / prices[0]) - 1.0
  bh_net = cost_model.net_after_tax(buy_hold_return)

  return BaselineResult(
    sma_window=sma_window,
    scenario=scenario,
    cost_model=cost_model,
    trades_count=len(trades),
    hit_rate=hit_rate,
    total_net_after_tax=total_net_after_tax,
    max_drawdown=max_dd,
    equity_curve=equity,
    buy_hold_return=buy_hold_return,
    buy_hold_net_after_tax=bh_net,
  )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def build_markdown_report(
  results: list[BaselineResult],
  meta: dict[str, object],
  years: float,
) -> str:
  """Build the trend baseline markdown report."""
  lines: list[str] = [
    "# BTC Trend Baseline (SMA Crossover) Analysis",
    "",
    "## Provenance & Dataset Information",
    "",
    f"- **Data source**: {meta.get('source', 'Coinbase Exchange REST API')}",
    f"- **Product**: {meta.get('product', 'BTC-USD')}",
    f"- **Granularity**: {meta.get('granularity', 86400)} s (daily OHLC)",
    f"- **Fetched at (UTC)**: {meta.get('fetched_at_utc', 'N/A')}",
    f"- **Rows (daily candles)**: {meta.get('rows', 'N/A')}",
    f"- **First date**: {meta.get('first_date', 'N/A')}",
    f"- **Last date**: {meta.get('last_date', 'N/A')}",
    f"- **Observation span**: {years:.2f} years",
    f"- **CSV SHA-256**: `{meta.get('sha256', 'N/A')}`",
    "",
    "## Model Assumptions and Limitations",
    "",
    "1. **Execution at Daily Close**: Entry and exit at daily Close.",
    "2. **Italian Taxation (33%)**: 33% on sum of gross trade returns; losses offset gains.",
    "   Fees are NOT deductible. This is an approximate, uncompounded tax model.",
    "3. **No Look-Ahead**: Signal at day t uses only data up to and including day t.",
    "4. **Compounding**: Equity curve compounds M*(1+r) per trade. Tax is applied once",
    "   on the sum of gross returns (approximate).",
    "5. **Configurations Tried**: 3 (SMA windows: 50, 100, 200 × 2 fee scenarios).",
    "",
    "## Results by SMA Window",
    "",
  ]

  for window in sorted(set(r.sma_window for r in results)):
    lines.extend([
      f"### SMA({window})",
      "",
      "| Scenario | Fee f | Extra e | Trades | Hit Rate | Total Net (Tax) | Max DD | Equity Final | B&H Return | B&H Net (Tax) |",
      "|---|---|---|---|---|---|---|---|---|---|",
    ])
    window_results = [r for r in results if r.sma_window == window]
    for res in window_results:
      sc = res.scenario
      eq_final = res.equity_curve[-1] if res.equity_curve else 1.0
      lines.append(
        f"| {sc.name} | {sc.f:.4f} | {sc.e:.4f} | {res.trades_count} | "
        f"{res.hit_rate:.2%} | {res.total_net_after_tax:+.4f} | "
        f"{res.max_drawdown:.2%} | {eq_final:.4f} | "
        f"{res.buy_hold_return:+.4f} | {res.buy_hold_net_after_tax:+.4f} |"
      )
    lines.append("")

  return "\n".join(lines) + "\n"


def write_csv_report(
  results: list[BaselineResult],
  path: Path,
) -> None:
  """Write trend baseline CSV report."""
  path.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = [
    "sma_window",
    "scenario",
    "fee_per_side_f",
    "extra_cost_e",
    "cost_multiplier_M",
    "trades_count",
    "hit_rate",
    "total_net_after_tax",
    "max_drawdown",
    "equity_final",
    "buy_hold_return",
    "buy_hold_net_after_tax",
    "note",
  ]
  import csv
  with path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for res in results:
      eq_final = res.equity_curve[-1] if res.equity_curve else 1.0
      writer.writerow({
        "sma_window": res.sma_window,
        "scenario": res.scenario.name,
        "fee_per_side_f": repr(res.scenario.f),
        "extra_cost_e": repr(res.scenario.e),
        "cost_multiplier_M": repr(res.cost_model.M),
        "trades_count": res.trades_count,
        "hit_rate": repr(res.hit_rate),
        "total_net_after_tax": repr(res.total_net_after_tax),
        "max_drawdown": repr(res.max_drawdown),
        "equity_final": repr(eq_final),
        "buy_hold_return": repr(res.buy_hold_return),
        "buy_hold_net_after_tax": repr(res.buy_hold_net_after_tax),
        "note": res.scenario.note,
      })


# ---------------------------------------------------------------------------
# CLI & Execution
# ---------------------------------------------------------------------------


def run_baseline(
  csv_path: Path,
  e_extra: float = DEFAULT_EXTRA_COST,
) -> tuple[list[BaselineResult], dict[str, object], float]:
  """Run complete trend baseline evaluation."""
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

  scenarios = get_baseline_scenarios(e_extra)
  results: list[BaselineResult] = []

  for sma_window in DEFAULT_SMAS:
    signals = compute_signals(close_prices, sma_window)

    for sc in scenarios:
      cm = CostModel(f=sc.f, e=sc.e)
      res = evaluate_baseline(close_prices, signals, cm, sc, sma_window)
      results.append(res)

  return results, meta, years


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
  parser = argparse.ArgumentParser(description="BTC Trend Baseline (SMA) Analysis.")
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
    default="research/reports/trend_baseline.md",
    help="Path for markdown report output (default: research/reports/trend_baseline.md)",
  )
  parser.add_argument(
    "--out-csv",
    default="research/reports/trend_baseline.csv",
    help="Path for CSV report output (default: research/reports/trend_baseline.csv)",
  )
  args = parser.parse_args()

  csv_path = Path(args.csv)
  if not csv_path.exists():
    print(f"ERROR: Input CSV {csv_path} does not exist", file=sys.stderr)
    sys.exit(1)

  results, meta, years = run_baseline(csv_path, e_extra=args.e)

  # Write reports
  md_report = build_markdown_report(results, meta, years)
  out_md_path = Path(args.out_md)
  out_md_path.parent.mkdir(parents=True, exist_ok=True)
  out_md_path.write_text(md_report, encoding="utf-8")

  out_csv_path = Path(args.out_csv)
  write_csv_report(results, out_csv_path)

  # Console summary
  print("=" * 78)
  print("BTC TREND BASELINE (SMA CROSSOVER) SUMMARY")
  print("=" * 78)
  print(f"Input CSV       : {csv_path}")
  print(f"Observation Span: {years:.2f} years")
  print(f"Markdown Report : {out_md_path}")
  print(f"CSV Report      : {out_csv_path}")
  print(f"Configurations tried: {len(results)}")
  print("-" * 78)
  print(f"{'SMA':<6} {'Scenario':<20} {'Trades':<8} {'HitRate':<10} {'Net(Tax)':<12} {'MaxDD':<10} {'Equity':<10} {'B&H Net':<12}")
  print("-" * 78)
  for res in results:
    eq_final = res.equity_curve[-1] if res.equity_curve else 1.0
    print(
      f"{res.sma_window:<6} "
      f"{res.scenario.name:<20} "
      f"{res.trades_count:<8} "
      f"{res.hit_rate:.2%}{'':<4} "
      f"{res.total_net_after_tax:+.4f}{'':<6} "
      f"{res.max_drawdown:.2%}{'':<4} "
      f"{eq_final:.4f}{'':<4} "
      f"{res.buy_hold_net_after_tax:+.4f}"
    )
  print("=" * 78)


if __name__ == "__main__":
  main()