# BTC Profitability Oracle & Data Pipeline

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
python -m scripts.oracle_breakeven \
  --csv research/data/btc_usd_1d.csv \
  --e 0.0015 \
  --out-md research/reports/oracle_breakeven.md \
  --out-csv research/reports/oracle_breakeven.csv
```

### Running Tests
```bash
python -m pytest tests/test_download_daily_ohlc.py tests/test_oracle_breakeven.py -v
```

## Cost and Tax Formulas

- **Cost multiplier**: $M = (1 - f)^2 \times (1 - e)$
  - $f$: maker or taker fee per side (entry and exit)
  - $e$: round-trip spread and slippage (default 0.15% = 15 bps)
- **Round-trip cost**: $c = 1 - M$
- **Break-even return (pre-tax)**: $r_{be,pre} = \frac{1}{M} - 1$
- **Break-even return (after-tax)**: $r_{be,tax} = \frac{1 - M}{M - 0.33}$
- **Single-trade net after tax**: $\text{net} = M \times (1 + r) - 1 - 0.33 \times \max(0, r)$
- **Required hit rate approximation**:
  - Pre-tax: $p^*_{pre} = 0.5 + \frac{c}{2 \cdot m}$
  - After-tax: $p^*_{tax} = 0.5 + \frac{c}{1.34 \cdot m}$
  *(where $m = \text{mean}(|r|)$)*

## Important Assumptions & Limitations

1. **Close-to-Close Execution**: Trades enter and exit at the daily Close. In practice,
   market execution occurs at current order book quotes with liquidity constraints.
2. **Italian Taxation**: 33% substitute tax on gross price appreciation. Exchange fees
   are assumed non-deductible (pending consultation with an Italian tax professional).
3. **Loss Offsetting**: Within-year gross gains and losses offset each other.
