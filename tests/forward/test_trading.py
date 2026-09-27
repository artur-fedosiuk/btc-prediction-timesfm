from evaluation.trading.backtest_engine import run_backtest
from evaluation.trading.signal import generate_signals
from evaluation.trading.types import PredictionRecord, StrategyConfig, TradingCosts


def pred(identity, time, forecast=110):
  return PredictionRecord(identity, time, 100, forecast, 90, 120, 120)


def test_no_future_equity():
  data = [
    pred("a", "2026-09-27T09:00:00Z"),
    pred("b", "2026-09-27T10:00:00Z"),
    pred("c", "2026-09-28T09:00:00Z"),
  ]
  result = run_backtest(
    data,
    StrategyConfig(accounting_mode="INVESTABLE_PORTFOLIO"),
    TradingCosts(0, 0, 0, 0),
  )
  assert [t.position_size for t in result.trades] == [1000, 1000, 1020]


def test_independent_fixed_notional():
  result = run_backtest(
    [pred("a", "2026-09-27T09:00:00Z"), pred("b", "2026-09-28T10:00:00Z")],
    trading_costs=TradingCosts(0, 0, 0, 0),
  )
  assert [t.position_size for t in result.trades] == [1000, 1000]


def test_offset_overlap():
  signals = generate_signals(
    [pred("a", "2026-09-16T12:00:00Z"), pred("b", "2026-09-17T13:00:00+02:00")],
    StrategyConfig(position_mode="single"),
  )
  assert [s.direction for s in signals] == ["LONG", "NO_TRADE"]


def test_flat_zero_threshold():
  assert (
    generate_signals(
      [pred("a", "2026-09-27T09:00:00Z", 100)], StrategyConfig(threshold_pct=0)
    )[0].direction
    == "NO_TRADE"
  )
