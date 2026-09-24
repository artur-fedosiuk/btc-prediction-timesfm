# Bitcoin (BTC/USD) Prediction Evaluation Report

> ⚠️ **Disclaimer**: This is a technical experiment, not financial advice. Past model performance does not predict future results.

## Experiment Configuration

- **Start date**: 2026-09-16
- **Duration**: 30 days
- **Prediction method(s)**: timesfm
- **Report generated**: 2026-09-24 16:56 UTC
- **Timezone**: UTC
- **Price verification**: TWAP 24h (CoinGecko + Binance cross-verified)
- **Actual price method**: TWAP — simple average of all hourly prices in the 24h window following each prediction

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total predictions | 10 |
| Verified predictions | 9 |
| MAE (Mean Absolute Error) | $587.53 |
| RMSE (Root Mean Square Error) | $723.55 |
| Mean Percentage Error | 0.73% |
| Direction Accuracy | 66.7% |
| Within Range Accuracy | 33.3% |

## Daily Results

| Date (UTC) | Initial ($) | Predicted ($) | Min ($) | Max ($) | Actual ($) | Error ($) | Error (%) | Dir. Pred. | Dir. Real | Dir. ✓ | In Range |
|------------|-------------|---------------|---------|---------|------------|-----------|-----------|------------|-----------|--------|----------|
| 2026-09-16 | 75,635.00 | 75,663.41 | 75,429.05 | 75,930.81 | 76,207.36 | 543.95 | 0.71% | 📈 up | 📈 up | ✅ | ❌ |
| 2026-09-16 | 75,768.00 | 75,775.91 | 75,548.17 | 76,018.97 | 76,247.32 | 471.41 | 0.62% | 📈 up | 📈 up | ✅ | ❌ |
| 2026-09-17 | 76,572.00 | 76,597.41 | 76,354.02 | 76,847.76 | 77,569.96 | 972.55 | 1.25% | 📈 up | 📈 up | ✅ | ❌ |
| 2026-09-18 | 80,780.00 | 80,726.36 | 80,103.88 | 81,123.00 | 81,184.39 | 458.03 | 0.56% | 📉 down | 📈 up | ❌ | ❌ |
| 2026-09-19 | 81,654.00 | 81,623.90 | 81,369.55 | 81,812.25 | 80,824.83 | 799.07 | 0.99% | 📉 down | 📉 down | ✅ | ❌ |
| 2026-09-20 | 80,807.00 | 80,795.17 | 80,551.83 | 81,023.90 | 82,313.81 | 1,518.64 | 1.84% | 📉 down | 📈 up | ❌ | ❌ |
| 2026-09-21 | 85,865.00 | 85,893.63 | 85,564.50 | 86,143.84 | 86,070.67 | 177.04 | 0.21% | 📈 up | 📈 up | ✅ | ✅ |
| 2026-09-22 | 86,189.00 | 86,206.48 | 85,933.06 | 86,493.86 | 86,061.21 | 145.27 | 0.17% | 📈 up | 📉 down | ❌ | ✅ |
| 2026-09-23 | 84,279.00 | 84,267.24 | 83,887.73 | 84,704.88 | 84,065.44 | 201.80 | 0.24% | 📉 down | 📉 down | ✅ | ✅ |
| 2026-09-24 | 84,593.00 | 84,510.38 | 84,019.73 | 85,015.66 | — | — | — | 📉 down | — | — | — |

## Price Source Verification (TWAP)

> Each "Actual" price is a **TWAP** (Time-Weighted Average Price): the simple average of all hourly BTC/USD prices in the 24h window after each prediction. Both CoinGecko and Binance TWAPs are computed independently; when they agree (≤1% discrepancy), the average of the two TWAPs is used.

| Date (UTC) | CoinGecko ($) | Binance ($) | Used ($) | Confidence |
|------------|---------------|-------------|---------|------------|
| 2026-09-16 | 76,207.36 | — | 76,207.36 | 🟡 medium |
| 2026-09-16 | 76,247.32 | — | 76,247.32 | 🟡 medium |
| 2026-09-17 | 77,569.96 | — | 77,569.96 | 🟡 medium |
| 2026-09-18 | 81,184.39 | — | 81,184.39 | 🟡 medium |
| 2026-09-19 | 80,824.83 | — | 80,824.83 | 🟡 medium |
| 2026-09-20 | 82,313.81 | — | 82,313.81 | 🟡 medium |
| 2026-09-21 | 86,070.67 | — | 86,070.67 | 🟡 medium |
| 2026-09-22 | 86,061.21 | — | 86,061.21 | 🟡 medium |
| 2026-09-23 | 84,065.44 | — | 84,065.44 | 🟡 medium |

## Conclusion

After 9 verified predictions, the model achieved a Mean Absolute Error of $587.53 (0.73% average percentage error). Direction was predicted correctly 66.7% of the time, and 33.3% of actual prices fell within the predicted range.

**Note**: These results are from a limited 30-day experiment. Cryptocurrency markets are highly volatile and unpredictable. This evaluation is for research purposes only and should not be used for trading decisions.
