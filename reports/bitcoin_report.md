# Bitcoin (BTC/USD) Prediction Evaluation Report

> ⚠️ **Disclaimer**: This is a technical experiment, not financial advice. Past model performance does not predict future results.

## Experiment Configuration

- **Start date**: 2026-09-16
- **Duration**: 30 days
- **Prediction method(s)**: timesfm
- **Report generated**: 2026-09-19 15:45 UTC
- **Timezone**: UTC
- **Price verification**: TWAP 24h (CoinGecko + Binance cross-verified)
- **Actual price method**: TWAP — simple average of all hourly prices in the 24h window following each prediction

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total predictions | 5 |
| Verified predictions | 3 |
| MAE (Mean Absolute Error) | $662.64 |
| RMSE (Root Mean Square Error) | $698.56 |
| Mean Percentage Error | 0.86% |
| Direction Accuracy | 100.0% |
| Within Range Accuracy | 0.0% |

## Daily Results

| Date (UTC) | Initial ($) | Predicted ($) | Min ($) | Max ($) | Actual ($) | Error ($) | Error (%) | Dir. Pred. | Dir. Real | Dir. ✓ | In Range |
|------------|-------------|---------------|---------|---------|------------|-----------|-----------|------------|-----------|--------|----------|
| 2026-09-16 | 75,635.00 | 75,663.41 | 75,429.05 | 75,930.81 | 76,207.36 | 543.95 | 0.71% | 📈 up | 📈 up | ✅ | ❌ |
| 2026-09-16 | 75,768.00 | 75,775.91 | 75,548.17 | 76,018.97 | 76,247.32 | 471.41 | 0.62% | 📈 up | 📈 up | ✅ | ❌ |
| 2026-09-17 | 76,572.00 | 76,597.41 | 76,354.02 | 76,847.76 | 77,569.96 | 972.55 | 1.25% | 📈 up | 📈 up | ✅ | ❌ |
| 2026-09-18 | 80,780.00 | 80,726.36 | 80,103.88 | 81,123.00 | — | — | — | 📉 down | — | — | — |
| 2026-09-19 | 81,654.00 | 81,623.90 | 81,369.55 | 81,812.25 | — | — | — | 📉 down | — | — | — |

## Price Source Verification (TWAP)

> Each "Actual" price is a **TWAP** (Time-Weighted Average Price): the simple average of all hourly BTC/USD prices in the 24h window after each prediction. Both CoinGecko and Binance TWAPs are computed independently; when they agree (≤1% discrepancy), the average of the two TWAPs is used.

| Date (UTC) | CoinGecko ($) | Binance ($) | Used ($) | Confidence |
|------------|---------------|-------------|---------|------------|
| 2026-09-16 | 76,207.36 | — | 76,207.36 | 🟡 medium |
| 2026-09-16 | 76,247.32 | — | 76,247.32 | 🟡 medium |
| 2026-09-17 | 77,569.96 | — | 77,569.96 | 🟡 medium |

## Conclusion

With only 3 verified prediction(s), it is too early to draw meaningful conclusions about the model's forecasting ability. The experiment will continue to collect data over the full 30-day window.
