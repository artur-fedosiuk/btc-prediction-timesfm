# BTC Trend Baseline (SMA Crossover) Analysis

## Provenance & Dataset Information

- **Data source**: Coinbase Exchange REST API
- **Product**: BTC-USD
- **Granularity**: 86400 s (daily OHLC)
- **Fetched at (UTC)**: 2026-10-12T10:01:09.804715+00:00
- **Rows (daily candles)**: 1825
- **First date**: 2021-10-13
- **Last date**: 2026-10-11
- **Observation span**: 5.00 years
- **CSV SHA-256**: `d05050f1ecd9d531414499faeb1a48890743760eaeecb509407c0cbb0f15a17e`

## Model Assumptions and Limitations

1. **Execution at Daily Close**: Entry and exit at daily Close.
2. **Italian Taxation (33%)**: 33% on sum of gross trade returns; losses offset gains.
   Fees are NOT deductible. This is an approximate, uncompounded tax model.
3. **No Look-Ahead**: Signal at day t uses only data up to and including day t.
4. **Compounding**: Equity curve compounds M*(1+r) per trade. Tax is applied once
   on the sum of gross returns (approximate).
5. **Configurations Tried**: 3 (SMA windows: 50, 100, 200 × 2 fee scenarios).

## Results by SMA Window

### SMA(50)

| Scenario | Fee f | Extra e | Trades | Hit Rate | Total Net (Tax) | Max DD | Equity Final | B&H Return | B&H Net (Tax) |
|---|---|---|---|---|---|---|---|---|---|
| coinbase_eu_maker | 0.0025 | 0.0015 | 53 | 18.87% | +0.6400 | 53.37% | 1.9937 | +0.4586 | +0.2978 |
| coinbase_eu_taker | 0.0050 | 0.0015 | 53 | 18.87% | +0.3690 | 56.09% | 1.5281 | +0.4586 | +0.2906 |

### SMA(100)

| Scenario | Fee f | Extra e | Trades | Hit Rate | Total Net (Tax) | Max DD | Equity Final | B&H Return | B&H Net (Tax) |
|---|---|---|---|---|---|---|---|---|---|
| coinbase_eu_maker | 0.0025 | 0.0015 | 33 | 21.21% | +0.8471 | 22.48% | 2.3563 | +0.4586 | +0.2978 |
| coinbase_eu_taker | 0.0050 | 0.0015 | 33 | 18.18% | +0.6750 | 25.54% | 1.9966 | +0.4586 | +0.2906 |

### SMA(200)

| Scenario | Fee f | Extra e | Trades | Hit Rate | Total Net (Tax) | Max DD | Equity Final | B&H Return | B&H Net (Tax) |
|---|---|---|---|---|---|---|---|---|---|
| coinbase_eu_maker | 0.0025 | 0.0015 | 17 | 35.29% | +1.0142 | 11.84% | 3.2500 | +0.4586 | +0.2978 |
| coinbase_eu_taker | 0.0050 | 0.0015 | 17 | 35.29% | +0.9212 | 13.59% | 2.9842 | +0.4586 | +0.2906 |

