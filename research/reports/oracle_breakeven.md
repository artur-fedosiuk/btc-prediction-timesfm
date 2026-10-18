# BTC Oracle Break-Even Analysis

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

1. **Execution at Daily Close**: Entry and exit are assumed exactly at daily Close.
   *Limitation*: Known OPTIMISTIC simplification; in live trading, execution occurs
   at the bid/ask spread and with execution delay.
2. **Italian Taxation (33%)**: Capital gains tax is modeled at 33% on gross price gains.
   *Limitation*: Fees and trading costs are treated as NON-DEDUCTIBLE.
   *Notice*: Assumption to be confirmed by an Italian qualified accountant.
3. **Tax Aggregation**: For multi-period sums, single-trade tax equals aggregated tax
   because the oracle trades exclusively periods with positive gains (r > threshold > 0).
   All losses offset gains within the year (optimistic simplification).
4. **Oracle Capital**: Independent trades sum net after-tax returns assuming a constant
   capital of 1 unit per trade (uncompounded arithmetic sum / years).
5. **Comparability**: The oracle sum per year and Buy-and-Hold per year are strictly
   comparable only as 'total return on initial capital 1'.
6. **Break-Even Hit Rate ($p^*$)**: APPROXIMATION assuming symmetric moves of mean size
   $m = \text{mean}(|r|)$, linearized costs, and loss offsetting.

## Empirical Return Distributions by Horizon (Overlapping Set)

| Horizon (days) | Samples (Overlapping) | Samples (Independent) | Mean(|r|) | Q10 | Q25 | Q50 (Median) | Q75 | Q90 | Q95 |
|---|---|---|---|---|---|---|---|---|---|
| 1d | 1824 | 1824 | 0.0184 | -0.0282 | -0.0121 | -0.0001 | +0.0128 | +0.0294 | +0.0444 |
| 3d | 1822 | 608 | 0.0330 | -0.0494 | -0.0232 | +0.0016 | +0.0248 | +0.0547 | +0.0759 |
| 7d | 1818 | 260 | 0.0508 | -0.0742 | -0.0331 | +0.0012 | +0.0386 | +0.0881 | +0.1209 |

## Oracle Results by Horizon

### Horizon 1 Day(s)

| Scenario | Fee f | Extra e | c (Round-trip) | $r_{be,pre}$ | $r_{be,tax}$ | $p^*_{pre}$ | $p^*_{tax}$ | Tradable % (Tax) | Mean Net (Tax) | Med Net (Tax) | Oracle Indep Sum/Yr (Tax) | B&H Net/Yr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coinbase_eu_maker | 0.0025 | 0.0015 | 0.0065 | 0.0065 | 0.0098 | 0.6759 | 0.7625 | 29.6% | 0.0128 | 0.0084 | +1.3843 | +0.0596 |
| coinbase_eu_taker | 0.0050 | 0.0015 | 0.0115 | 0.0116 | 0.0174 | 0.8107 | 0.9638 | 19.1% | 0.0134 | 0.0088 | +0.9332 | +0.0581 |
| kraken_t1_maker | 0.0040 | 0.0015 | 0.0095 | 0.0096 | 0.0143 | 0.7568 | 0.8833 | 22.7% | 0.0132 | 0.0087 | +1.0899 | +0.0587 |
| kraken_t1_taker | 0.0080 | 0.0015 | 0.0174 | 0.0177 | 0.0267 | 0.9721 | unreachable | 12.0% | 0.0134 | 0.0090 | +0.5861 | +0.0564 |
| kraken_t2_maker | 0.0030 | 0.0015 | 0.0075 | 0.0075 | 0.0113 | 0.7029 | 0.8028 | 27.6% | 0.0127 | 0.0081 | +1.2775 | +0.0593 |
| kraken_t2_taker | 0.0060 | 0.0015 | 0.0134 | 0.0136 | 0.0205 | 0.8646 | unreachable | 16.6% | 0.0132 | 0.0090 | +0.7987 | +0.0576 |
| repo_assumption | 0.0018 | 0.0000 | 0.0035 | 0.0035 | 0.0052 | 0.5948 | 0.6415 | 37.4% | 0.0129 | 0.0084 | +1.7608 | +0.0605 |

### Horizon 3 Day(s)

| Scenario | Fee f | Extra e | c (Round-trip) | $r_{be,pre}$ | $r_{be,tax}$ | $p^*_{pre}$ | $p^*_{tax}$ | Tradable % (Tax) | Mean Net (Tax) | Med Net (Tax) | Oracle Indep Sum/Yr (Tax) | B&H Net/Yr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coinbase_eu_maker | 0.0025 | 0.0015 | 0.0065 | 0.0065 | 0.0098 | 0.5981 | 0.6465 | 38.5% | 0.0212 | 0.0149 | +0.9939 | +0.0596 |
| coinbase_eu_taker | 0.0050 | 0.0015 | 0.0115 | 0.0116 | 0.0174 | 0.6734 | 0.7588 | 28.9% | 0.0222 | 0.0165 | +0.7830 | +0.0581 |
| kraken_t1_maker | 0.0040 | 0.0015 | 0.0095 | 0.0096 | 0.0143 | 0.6433 | 0.7139 | 31.7% | 0.0223 | 0.0166 | +0.8605 | +0.0587 |
| kraken_t1_taker | 0.0080 | 0.0015 | 0.0174 | 0.0177 | 0.0267 | 0.7634 | 0.8932 | 22.7% | 0.0213 | 0.0153 | +0.5870 | +0.0564 |
| kraken_t2_maker | 0.0030 | 0.0015 | 0.0075 | 0.0075 | 0.0113 | 0.6132 | 0.6690 | 36.3% | 0.0214 | 0.0153 | +0.9465 | +0.0593 |
| kraken_t2_taker | 0.0060 | 0.0015 | 0.0134 | 0.0136 | 0.0205 | 0.7034 | 0.8036 | 26.3% | 0.0222 | 0.0154 | +0.7121 | +0.0576 |
| repo_assumption | 0.0018 | 0.0000 | 0.0035 | 0.0035 | 0.0052 | 0.5529 | 0.5790 | 45.2% | 0.0210 | 0.0151 | +1.1537 | +0.0605 |

### Horizon 7 Day(s)

| Scenario | Fee f | Extra e | c (Round-trip) | $r_{be,pre}$ | $r_{be,tax}$ | $p^*_{pre}$ | $p^*_{tax}$ | Tradable % (Tax) | Mean Net (Tax) | Med Net (Tax) | Oracle Indep Sum/Yr (Tax) | B&H Net/Yr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coinbase_eu_maker | 0.0025 | 0.0015 | 0.0065 | 0.0065 | 0.0098 | 0.5638 | 0.5952 | 44.6% | 0.0339 | 0.0243 | +0.7870 | +0.0596 |
| coinbase_eu_taker | 0.0050 | 0.0015 | 0.0115 | 0.0116 | 0.0174 | 0.6127 | 0.6682 | 39.2% | 0.0329 | 0.0219 | +0.6716 | +0.0581 |
| kraken_t1_maker | 0.0040 | 0.0015 | 0.0095 | 0.0096 | 0.0143 | 0.5932 | 0.6390 | 40.8% | 0.0338 | 0.0231 | +0.7159 | +0.0587 |
| kraken_t1_taker | 0.0080 | 0.0015 | 0.0174 | 0.0177 | 0.0267 | 0.6712 | 0.7556 | 33.1% | 0.0321 | 0.0208 | +0.5527 | +0.0564 |
| kraken_t2_maker | 0.0030 | 0.0015 | 0.0075 | 0.0075 | 0.0113 | 0.5736 | 0.6098 | 43.5% | 0.0337 | 0.0239 | +0.7626 | +0.0593 |
| kraken_t2_taker | 0.0060 | 0.0015 | 0.0134 | 0.0136 | 0.0205 | 0.6322 | 0.6974 | 37.3% | 0.0324 | 0.0212 | +0.6293 | +0.0576 |
| repo_assumption | 0.0018 | 0.0000 | 0.0035 | 0.0035 | 0.0052 | 0.5344 | 0.5513 | 46.5% | 0.0356 | 0.0256 | +0.8616 | +0.0605 |

